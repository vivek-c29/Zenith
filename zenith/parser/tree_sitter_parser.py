"""
Multi-language code parser using tree-sitter.
Extracts symbols (functions, classes, methods, imports) and produces
structural code chunks for embedding and LLM context.
"""

import hashlib
import os
from typing import List, Optional

from loguru import logger
from tree_sitter_language_pack import get_parser

from zenith.utils.text_utils import count_tokens
from .language_map import detect_language
from .models import CodeChunk, CodeSymbol, ParsedFile, SymbolKind


# AST node types that represent extractable symbols, per language family.
# Each entry maps a tree-sitter node type to our SymbolKind.
SYMBOL_NODE_TYPES: dict[str, dict[str, SymbolKind]] = {
    "python": {
        "function_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
        "import_statement": SymbolKind.IMPORT,
        "import_from_statement": SymbolKind.IMPORT,
    },
    "javascript": {
        "function_declaration": SymbolKind.FUNCTION,
        "arrow_function": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "import_statement": SymbolKind.IMPORT,
    },
    "typescript": {
        "function_declaration": SymbolKind.FUNCTION,
        "arrow_function": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "import_statement": SymbolKind.IMPORT,
    },
    "java": {
        "method_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "interface_declaration": SymbolKind.CLASS,
        "import_declaration": SymbolKind.IMPORT,
    },
    "go": {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.CLASS,
        "import_declaration": SymbolKind.IMPORT,
    },
    "rust": {
        "function_item": SymbolKind.FUNCTION,
        "impl_item": SymbolKind.CLASS,
        "struct_item": SymbolKind.CLASS,
        "enum_item": SymbolKind.CLASS,
        "use_declaration": SymbolKind.IMPORT,
    },
    "cpp": {
        "function_definition": SymbolKind.FUNCTION,
        "class_specifier": SymbolKind.CLASS,
        "preproc_include": SymbolKind.IMPORT,
    },
    "c": {
        "function_definition": SymbolKind.FUNCTION,
        "struct_specifier": SymbolKind.CLASS,
        "preproc_include": SymbolKind.IMPORT,
    },
    "ruby": {
        "method": SymbolKind.FUNCTION,
        "class": SymbolKind.CLASS,
        "module": SymbolKind.CLASS,
    },
}

# Fallback node types for languages not explicitly listed
DEFAULT_SYMBOL_NODE_TYPES: dict[str, SymbolKind] = {
    "function_definition": SymbolKind.FUNCTION,
    "function_declaration": SymbolKind.FUNCTION,
    "method_declaration": SymbolKind.METHOD,
    "class_definition": SymbolKind.CLASS,
    "class_declaration": SymbolKind.CLASS,
    "import_statement": SymbolKind.IMPORT,
    "import_declaration": SymbolKind.IMPORT,
}


class CodeParser:
    """
    Multi-language code parser that uses tree-sitter to extract structured
    symbols and produce code chunks for embedding.
    """

    def __init__(self):
        self._parser_cache: dict[str, object] = {}

    def _get_parser(self, language: str):
        """Get or create a cached tree-sitter parser for the given language."""
        if language not in self._parser_cache:
            try:
                self._parser_cache[language] = get_parser(language)
                logger.debug(f"Loaded tree-sitter parser for: {language}")
            except Exception as e:
                logger.error(f"Failed to load parser for {language}: {e}")
                raise
        return self._parser_cache[language]

    def _get_symbol_types(self, language: str) -> dict[str, SymbolKind]:
        """Get the symbol node type mapping for a language."""
        return SYMBOL_NODE_TYPES.get(language, DEFAULT_SYMBOL_NODE_TYPES)

    def _extract_name(self, node, source_bytes: bytes) -> str:
        """Extract the name/identifier from an AST node."""
        # Look for a direct 'name' child (common in most languages)
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            # For Python decorated definitions
            if child.type == "dotted_name":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

        # For import statements, return the full text
        if "import" in node.type:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()

        return "<anonymous>"

    def _extract_signature(self, node, source_bytes: bytes, language: str) -> Optional[str]:
        """Extract the function/method signature (first line up to the body)."""
        if "import" in node.type:
            return None

        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        # Return just the first line as signature
        first_line = text.split("\n")[0].strip()
        return first_line

    def _extract_docstring(self, node, source_bytes: bytes, language: str) -> Optional[str]:
        """Extract docstring from a function/class node if present."""
        if language == "python":
            # In Python, the docstring is inside the 'block' child.
            # It can appear as a direct 'string' node or wrapped in 'expression_statement'.
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        # Case 1: Direct string node (tree-sitter >= 0.23)
                        if block_child.type == "string":
                            doc = source_bytes[block_child.start_byte:block_child.end_byte].decode(
                                "utf-8", errors="replace"
                            )
                            return doc.strip('"\' \n')
                        # Case 2: Wrapped in expression_statement
                        if block_child.type == "expression_statement":
                            for expr_child in block_child.children:
                                if expr_child.type == "string":
                                    doc = source_bytes[expr_child.start_byte:expr_child.end_byte].decode(
                                        "utf-8", errors="replace"
                                    )
                                    return doc.strip('"\' \n')
                            break  # Only check the first expression_statement
                        # If first meaningful child isn't a string, no docstring
                        if block_child.type not in ("newline", "indent", "comment", "NEWLINE"):
                            break
                    break

        elif language in ("javascript", "typescript", "java", "go", "rust", "cpp", "c"):
            # Check for a comment node immediately before this node
            if node.prev_sibling and node.prev_sibling.type in ("comment", "block_comment"):
                doc = source_bytes[node.prev_sibling.start_byte:node.prev_sibling.end_byte].decode(
                    "utf-8", errors="replace"
                )
                return doc.strip("/* \n")

        return None

    def _walk_tree(
        self,
        node,
        source_bytes: bytes,
        source_lines: List[str],
        language: str,
        symbol_types: dict[str, SymbolKind],
        parent_class: Optional[str] = None,
    ) -> List[CodeSymbol]:
        """Recursively walk the AST and extract symbols."""
        symbols: List[CodeSymbol] = []

        for child in node.children:
            node_type = child.type

            if node_type in symbol_types:
                kind = symbol_types[node_type]
                name = self._extract_name(child, source_bytes)

                # If this is a function inside a class, it's a method
                actual_kind = kind
                if kind == SymbolKind.FUNCTION and parent_class is not None:
                    actual_kind = SymbolKind.METHOD

                # Line numbers are 0-indexed in tree-sitter, convert to 1-indexed
                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1

                source_code = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )

                symbol = CodeSymbol(
                    name=name,
                    kind=actual_kind,
                    start_line=start_line,
                    end_line=end_line,
                    signature=self._extract_signature(child, source_bytes, language),
                    docstring=self._extract_docstring(child, source_bytes, language),
                    source_code=source_code,
                    parent_class=parent_class,
                )
                symbols.append(symbol)

                # If this is a class, recurse into it to find methods
                if kind == SymbolKind.CLASS and "import" not in node_type:
                    class_name = name
                    nested = self._walk_tree(
                        child, source_bytes, source_lines, language, symbol_types, parent_class=class_name
                    )
                    symbols.extend(nested)
            else:
                # Continue walking non-symbol nodes to find nested symbols
                nested = self._walk_tree(
                    child, source_bytes, source_lines, language, symbol_types, parent_class=parent_class
                )
                symbols.extend(nested)

        return symbols

    def parse_source(self, source_code: str, language: str, file_path: str = "<string>") -> ParsedFile:
        """
        Parse raw source code string and extract symbols.

        Args:
            source_code: The source code to parse.
            language: Tree-sitter language identifier (e.g. 'python', 'javascript').
            file_path: Optional file path for metadata.

        Returns:
            ParsedFile with extracted symbols and chunks.
        """
        parser = self._get_parser(language)
        source_bytes = source_code.encode("utf-8")
        source_lines = source_code.split("\n")

        tree = parser.parse(source_bytes)
        root = tree.root_node

        symbol_types = self._get_symbol_types(language)
        symbols = self._walk_tree(root, source_bytes, source_lines, language, symbol_types)

        # Extract imports separately as strings
        imports = [s.source_code for s in symbols if s.kind == SymbolKind.IMPORT]

        logger.info(
            f"Parsed {file_path} ({language}): "
            f"{len(symbols)} symbols, {len(imports)} imports, {len(source_lines)} lines"
        )

        parsed = ParsedFile(
            file_path=file_path,
            language=language,
            symbols=symbols,
            imports=imports,
            total_lines=len(source_lines),
        )

        # Generate structural chunks
        parsed.chunks = self._generate_chunks(parsed, source_code)

        return parsed

    def parse_file(self, file_path: str) -> Optional[ParsedFile]:
        """
        Parse a source file from disk.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            ParsedFile with extracted symbols and chunks, or None if language
            is unsupported.
        """
        language = detect_language(file_path)
        if language is None:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

        return self.parse_source(source_code, language, file_path=file_path)

    def _extract_calls_from_text(self, text: str, language: str) -> List[str]:
        """Parse a snippet of code and extract all function/method calls."""
        try:
            parser = self._get_parser(language)
            code_bytes = text.encode("utf-8")
            tree = parser.parse(code_bytes)
            
            calls = set()
            def walk(node):
                if "call" in node.type:
                    if len(node.children) > 0:
                        func_node = node.children[0]
                        if func_node.type in ("attribute", "member_expression") and len(func_node.children) >= 3:
                            name_node = func_node.children[-1]
                            calls.add(code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8"))
                        elif func_node.type == "identifier":
                            calls.add(code_bytes[func_node.start_byte:func_node.end_byte].decode("utf-8"))
                for child in node.children:
                    walk(child)
            
            walk(tree.root_node)
            return list(calls)
        except Exception:
            return []

    def _generate_chunks(
        self,
        parsed: ParsedFile,
        source_code: str,
        max_tokens: int = 1500,
    ) -> List[CodeChunk]:
        """
        Generate structural code chunks from parsed symbols.

        Chunks are created along function/class boundaries. If a single
        symbol exceeds max_tokens, it is split into smaller sub-chunks.
        """
        chunks: List[CodeChunk] = []
        source_lines = source_code.split("\n")
        covered_lines: set[int] = set()

        # Create chunks from each non-import symbol
        non_import_symbols = [s for s in parsed.symbols if s.kind != SymbolKind.IMPORT]

        # Sort by start_line to process in order
        non_import_symbols.sort(key=lambda s: s.start_line)

        for symbol in non_import_symbols:
            # Skip methods — they are included as part of their parent class chunk
            if symbol.kind == SymbolKind.METHOD:
                continue

            content = symbol.source_code
            tokens = count_tokens(content)

            if tokens <= max_tokens:
                chunk_id = self._make_chunk_id(parsed.file_path, symbol.name, symbol.start_line)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed.file_path,
                        language=parsed.language,
                        symbol_name=symbol.name,
                        symbol_kind=symbol.kind,
                        content=content,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        calls=self._extract_calls_from_text(content, parsed.language),
                        token_count=tokens,
                    )
                )
            else:
                # Large symbol (e.g., big class) — split into sub-chunks
                sub_chunks = self._split_large_symbol(symbol, parsed, max_tokens)
                chunks.extend(sub_chunks)

            # Track which lines are covered
            for line_num in range(symbol.start_line, symbol.end_line + 1):
                covered_lines.add(line_num)

        # Create a chunk for top-level code not covered by any symbol
        # (module-level imports, constants, etc.)
        uncovered_lines = []
        for i, line in enumerate(source_lines, start=1):
            if i not in covered_lines and line.strip():
                uncovered_lines.append((i, line))

        if uncovered_lines:
            start_line = uncovered_lines[0][0]
            end_line = uncovered_lines[-1][0]
            content = "\n".join(line for _, line in uncovered_lines)
            tokens = count_tokens(content)

            if content.strip():
                chunk_id = self._make_chunk_id(parsed.file_path, "<module>", start_line)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed.file_path,
                        language=parsed.language,
                        symbol_name="<module>",
                        symbol_kind=None,
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                        calls=self._extract_calls_from_text(content, parsed.language),
                        token_count=tokens,
                    )
                )

        return chunks

    def _split_large_symbol(
        self,
        symbol: CodeSymbol,
        parsed: ParsedFile,
        max_tokens: int,
    ) -> List[CodeChunk]:
        """Split a large symbol (e.g., a big class) into sub-chunks by its child methods."""
        chunks: List[CodeChunk] = []

        # Find child methods of this class
        child_methods = [
            s for s in parsed.symbols
            if s.parent_class == symbol.name and s.kind == SymbolKind.METHOD
        ]

        if child_methods:
            for method in child_methods:
                content = method.source_code
                tokens = count_tokens(content)
                chunk_id = self._make_chunk_id(
                    parsed.file_path, f"{symbol.name}.{method.name}", method.start_line
                )
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed.file_path,
                        language=parsed.language,
                        symbol_name=f"{symbol.name}.{method.name}",
                        symbol_kind=SymbolKind.METHOD,
                        content=content,
                        start_line=method.start_line,
                        end_line=method.end_line,
                        calls=self._extract_calls_from_text(content, parsed.language),
                        token_count=tokens,
                    )
                )
        else:
            # No methods found — just split by line count
            lines = symbol.source_code.split("\n")
            chunk_size = 50  # lines per sub-chunk
            for i in range(0, len(lines), chunk_size):
                sub_lines = lines[i : i + chunk_size]
                content = "\n".join(sub_lines)
                tokens = count_tokens(content)
                start = symbol.start_line + i
                end = min(symbol.start_line + i + chunk_size - 1, symbol.end_line)
                chunk_id = self._make_chunk_id(parsed.file_path, symbol.name, start)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed.file_path,
                        language=parsed.language,
                        symbol_name=symbol.name,
                        symbol_kind=symbol.kind,
                        content=content,
                        start_line=start,
                        end_line=end,
                        calls=self._extract_calls_from_text(content, parsed.language),
                        token_count=tokens,
                    )
                )

        return chunks

    @staticmethod
    def _make_chunk_id(file_path: str, symbol_name: str, start_line: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{file_path}::{symbol_name}::{start_line}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
