#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import re
import json
from typing import List, Tuple, Optional

from .worker import Lean4ServerScheduler, LLMServerScheduler


class Sketch:
    llm_scheduler = None
    lean4_scheduler = None
    max_error_masking_rounds = 20

    @classmethod
    def launch_llmserver(cls, client_config: List[dict], max_llm_requests: int=128, name: str='Sketch', **kwargs):
        cls.llm_scheduler = LLMServerScheduler(
            client_config = client_config,
            max_running_requests_per_client = max_llm_requests,
            name = name,
            **kwargs,
        )
        
    @classmethod
    def close_llmserver(cls):
        if cls.llm_scheduler is not None:
            cls.llm_scheduler.close()
    
    @classmethod
    def launch_lean4server(cls, max_lean4_requests: int=1, verify_timeout: int=180, startup_timeout: int=None, max_error_masking_rounds: int=20, cwd: str='./mathlib4', name: str='sketch_verifier'):
        cls.max_error_masking_rounds = max_error_masking_rounds
        command = '''import Mathlib
import Aesop
set_option maxHeartbeats 0
syntax "prove_with" ("[" term,* "]")? : tactic
macro_rules
| `(tactic| prove_with [$args,*]) => `(tactic| sorry)
| `(tactic| prove_with)           => `(tactic| sorry)
'''
        cls.lean4_scheduler = Lean4ServerScheduler(
            max_concurrent_requests = max_lean4_requests, 
            timeout = verify_timeout, 
            startup_timeout = startup_timeout,
            memory_limit = 10,
            name = name,
            cwd = cwd, 
            command = json.dumps({"cmd": command}), 
            share_header = 'import Mathlib',
        )
        
    @classmethod
    def close_lean4server(cls):
        if cls.lean4_scheduler is not None:
            cls.lean4_scheduler.close()
        
    def error_masking(self, sketch: str) -> str:
        """Remove the erroneous lines until there are no more errors"""
        
        def get_error_line(code: str) -> Tuple[List[int], dict]:
            """Verify and return all erroneous lines."""
            # code = re.sub('import .*\n', '', code)
            request_id = self.lean4_scheduler.submit_request(dict(code=code, allTactics=False))
            repl_output = self.lean4_scheduler.get_request_outputs(request_id)
            error_line = []
            if repl_output['system_messages']:
                raise ValueError(repl_output['system_messages'])
            for error in repl_output.get('errors', []):
                if error['data'].startswith('unsolved goals'):
                    pass
                else:
                    error_line.append(error['pos']['line'])
            return error_line, repl_output
        
        def parse_lean_code(lines: List[str]) -> List[dict]:
            """Parse Lean code into a tree structure based on indentation"""
            nonlocal error_line
            stack = []
            tree = []

            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue
                else:
                    indent_level = len(line) - len(line.lstrip())
                node = {"content": clean_line, "children": [], "error": i+1 in error_line}

                if not stack:
                    tree.append(node)
                    stack.append((indent_level, node))
                else:
                    while stack and indent_level <= stack[-1][0]:
                        stack.pop()

                    if stack:
                        stack[-1][1]["children"].append(node)
                    else:
                        tree.append(node)
                    stack.append((indent_level, node))

            return tree
        
        def print_tree(tree: List[dict], level: int=0, comment: bool=False, if_sorry: bool=False, strict: bool=False) -> None:
            """
            Recursively prints a structured proof tree, replacing content with 'sorry' 
            for error nodes or subtrees based on configuration.

            Parameters:
                tree (List[dict]): The current list of nodes at this level of the proof tree.
                                Each node is a dict with keys: 'content' (str), 
                                'error' (bool), and 'children' (List[dict]).
                level (int): The current indentation level (used for visual formatting).
                comment (bool): Whether to comment out this and all child levels 
                                (typically set if the parent node had an error).
                if_sorry (bool): Indicates whether this subtree has already been replaced with 'sorry'.
                                If True, suppress further output within this branch.
                strict (bool): If True, any error in a child node will cause its parent 
                            to be replaced with 'sorry', even if the parent has no error.

            External (nonlocal) variables:
                ori_line_count (int): Counts total number of printed lines.
                error_line (int): Tracks the line where an error occurred (if used).
                formal_proof_sorry (List[str]): Stores the final output lines (indented strings).
                sorries (List[int]): Tracks line indices where 'sorry' was inserted.

            Behavior:
                - For deep levels (level > 1), directly replace error nodes or their parents with 'sorry'.
                - For top levels (level <= 1), optionally comment out or replace based on error presence.
                - Recursive processing continues unless the current branch has been 'sorry'-blocked.
            """
            nonlocal ori_line_count, error_line, formal_proof_sorry, sorries
            for node in tree:
                if level > 1:
                    # Deeper level: directly suppress error nodes with 'sorry'
                    if node["error"]:
                        if not if_sorry:
                            sorries.append(len(formal_proof_sorry))
                            formal_proof_sorry.append("  " * level + '-- ' * comment + "sorry")
                            if_sorry = True
                    else:
                        if not if_sorry:
                            # Check if any child has error to decide if we still print content
                            child_error = any(child["error"] for child in node["children"])
                            if strict and child_error:
                                formal_proof_sorry.append("  " * level + '-- ' * comment + "sorry")
                                if_sorry = True
                            else:
                                formal_proof_sorry.append("  " * level + '-- ' * comment + node["content"])
                    ori_line_count += 1
                    if node["children"]:
                        print_tree(node["children"], level + 1, comment, if_sorry, strict)
                        
                else:
                    # Top-level: decide whether to comment or replace with 'sorry'
                    if_comment = comment or node["error"]
                    if not if_sorry:
                        child_error = any(child["error"] for child in node["children"])
                        if strict and child_error:
                            formal_proof_sorry.append("  " * level + '-- ' * if_comment + "sorry")
                            if_sorry = True
                        else:
                            formal_proof_sorry.append("  " * level + '-- ' * if_comment + node["content"])
                    ori_line_count += 1
                    if node["children"]:
                        print_tree(node["children"], level + 1, if_comment, if_sorry, strict)
        
        # Split the initial sketch into lines
        formal_proof_sorry = sketch.split('\n')
        last_error_line = []
        count = 0
        
        while True:
            if count >= self.max_error_masking_rounds:
                print(
                    f"Sketch error masking reached max_error_masking_rounds={self.max_error_masking_rounds}; "
                    "continuing with the current masked sketch.",
                    flush=True,
                )
                break

            # Run REPL to get error line and output
            error_line, repl_output = get_error_line('\n'.join(formal_proof_sorry))
            tree = parse_lean_code(formal_proof_sorry)

            if not error_line:
                break
            
            # Reset state for this iteration
            ori_line_count = 1
            sorries = []
            formal_proof_sorry = []
            
            # Use strict mode if same error repeats
            if error_line == last_error_line and count > 1:
                print_tree(tree, strict=True)
            else:
                print_tree(tree)
                
            last_error_line = error_line
            count += 1
            
        # # Append 'sorry' if unsolved goals remain
        if any(error['data'].startswith('unsolved goals') for error in repl_output.get('errors', [])):
            formal_proof_sorry.append('  sorry')
            
        return '\n'.join(formal_proof_sorry)

    def prepare_header(self, header: str) -> str:
        """Prepare dataset headers as assumptions available to the target theorem."""

        def axiomize_sorry_decl(match: re.Match) -> str:
            declaration = match.group(2).rstrip()
            return f"axiom {declaration}\n"

        return re.sub(
            r'(?ms)^\s*(theorem|lemma)\s+(.+?)\s*:=\s*by\s+sorry\s*',
            axiomize_sorry_decl,
            header.strip(),
        )

    def compose_sketch(self, data: dict, target_sketch: str) -> str:
        """Combine the sanitized Lean context with the target theorem sketch."""
        header = self.prepare_header(data.get('header', ''))
        if not header:
            return target_sketch.strip()
        return f"{header}\n\n{target_sketch.strip()}"
        
    def llm_generator(self, data: dict, draft: str) -> str:
        """get raw output from llm"""
        # get what you need
        formal_statement = data['formal_statement']
        header = self.prepare_header(data['header'])
        informal_statement = data.get('informal_statement', '')
        informal_prefix = data.get('informal_prefix', '')
        dataset_draft = data.get('informal_proof', '')
        if isinstance(dataset_draft, str) and dataset_draft.strip():
            proof_draft = dataset_draft.strip()
            draft_source = 'pre-generated dataset informal_proof'
        else:
            proof_draft = draft.strip()
            draft_source = 'DSP+ Draft stage output'
        
        # get llm prompt
        prompt = f'''You are generating the DSP+ Sketch stage for one Lean 4 theorem.

The proof Draft below is a sequence of Markdown proof steps derived from an Isabelle proof. It should express formulas in Lean-like notation using target-side names from a proof-masked Lean 4 context. Treat it as a proof plan, not as executable Lean code. Ignore Markdown headings and inline backticks when translating it.

You must write Lean code for the TARGET THEOREM only. Do not repeat the Lean header, imports, classes, definitions, axioms, namespace commands, or prior theorem declarations in your output. The verifier will prepend the Lean header automatically.

Use only names and notation available in the Lean header below. In particular:
- Preserve the exact target theorem statement.
- Follow the Draft's proof strategy and step order, while combining steps when the header justifies a shorter direct Lean proof.
- Prefer the definitions, axioms, and prior facts named in the header.
- Do not invent Mathlib lemmas, algebraic structures, notation, or theorem names that do not appear in the header.
- Check every Draft identifier and formula against the header even when it already looks like Lean.
- If the header defines operations explicitly, use those names explicitly. For example, use `AddMonoid.add` instead of unprovided `+` notation.
- For a legacy Draft, translate source-side aliases and notation into exact header names. A name such as `foo_def` may mean "unfold `foo`", and a symbol such as `+C` may mean `AddMonoid.add`; never emit either form unless it actually appears in the header.
- Match each cited Draft fact to an available declaration by its statement as well as its approximate name. If the Draft says to apply `integral_shift` and that axiom is present, use the header's exact declaration and argument order.
- Prefer the shortest direct Lean proof justified by the Draft and header. For example, a Draft that says to unfold `circleAverage` and `circleMap`, obtains `integral (fun theta => f (AddMonoid.add theta c)) = integral f`, and applies `integral_shift f c` should become `unfold circleAverage circleMap` followed by `exact integral_shift f c`.
- If the proof needs an intermediate claim, introduce it with `have`.
- For any subclaim that should be solved later, use `by` followed by `prove_with[...]`.
- Do not copy Markdown headings, prose, or inline backticks into the Lean proof.
- Output exactly one Lean code block containing only the target theorem and its sketch proof.

informal_prefix:
{informal_prefix}

informal_statement:
{informal_statement}

proof_draft ({draft_source}):
{proof_draft}

Lean header available before the target theorem:
```lean4
{header}
```

Target theorem statement:
```lean4
{formal_statement}
```

Remember: output only the target theorem, not the header.
'''
    
        # get llm output
        request_id = self.llm_scheduler.submit_request(prompt)
        output = self.llm_scheduler.get_request_outputs(request_id)
        if not output:
            raise RuntimeError("Sketch LLM request failed")
        else:
            return output
        
    def extract_sketch(self, raw_sketch: str) -> str:
        """extract sketch from llm output"""
        sketch = raw_sketch.split('</think>')[-1].strip()
        sketch = re.search(r'```lean4?\n(.+?)\n```', sketch, re.DOTALL).group(1)
        sketch = re.sub(r'import .+?\n', '', sketch)
        sketch = sketch.strip()
        return sketch
    
    
    def judge_if_exist(self, res_dir: str, idx: int) -> bool:
        """judge whether the historical file exists"""
        return os.path.exists(os.path.join(res_dir, f"{idx}_sketch.json")) \
                or os.path.exists(os.path.join(res_dir, f"{idx}_prove.json"))
                
    def read_draft(self, res_dir: str, idx: int) -> Optional[str]:
        """got draft from file"""
        if os.path.exists(os.path.join(res_dir, f"{idx}_draft.json")):
            with open(os.path.join(res_dir, f"{idx}_draft.json"), 'r', encoding='utf-8') as fp:
                output = json.load(fp)
            return output['draft']
        else:
            return None
                
    def save_sketch(self, res_dir: str, idx: int, raw_sketch: str, sketch: str) -> None:
        """save sketch to file"""
        with open(f"{res_dir}/{idx}_draft.json", 'r', encoding='utf-8') as fpd, open(f"{res_dir}/{idx}_sketch.json", 'w', encoding='utf-8') as fps:
            prev_result = json.load(fpd)
            prev_result.update({
                "raw_sketch": raw_sketch,
                "sketch": sketch,
            })
            json.dump(prev_result, fps, indent=2, ensure_ascii=False)
        os.remove(f"{res_dir}/{idx}_draft.json")
                
    def run(self, data: dict, res_dir: str, idx: int) -> None:
        """main interface"""
        if not self.judge_if_exist(res_dir, idx):
            draft = self.read_draft(res_dir, idx)
            if draft is not None:
                print(f"   {data['name']} - {idx} sketching...", flush=True)
                raw_sketch = self.llm_generator(data, draft)
                extract_sketch = self.extract_sketch(raw_sketch)
                sketch = self.error_masking(self.compose_sketch(data, extract_sketch))
                self.save_sketch(res_dir, idx, raw_sketch, sketch)
                print(f"Done Sketch {data['name']} - {idx}", flush=True)
                
    def run_direct(self, data: dict, draft: str) -> str:
        """Interface that directly returns results"""
        print(f"   {data['name']} sketching...", flush=True)
        raw_sketch = self.llm_generator(data, draft)
        extract_sketch = self.extract_sketch(raw_sketch)
        sketch = self.error_masking(self.compose_sketch(data, extract_sketch))
        print(f"Done Sketch {data['name']}", flush=True)
        return sketch
