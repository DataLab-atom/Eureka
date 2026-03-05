
import re
import logging
from typing import Dict, List, Optional, Tuple

from .judge.agent_as_a_judge.text_eval_agent import TextEvalAgent
from .mas.text_agents import TextModifierAgent

SectionKey = str

def extract_sections(latex: str) -> Dict[SectionKey, Tuple[str, Tuple[int, int]]]:
    """Extract target sections from a LaTeX file.

    Returns mapping: key -> (body_text_without_header, (start_idx, end_idx))
    Keys: introduction, related_work, method, experiment
    """
    # Patterns for section headers (accept starred or not)
    pat_map = {
        "introduction": r"\\section\*?\{[Ii]ntroduction\}",
        "related_work": r"\\section\*?\{[Rr]elated\s+[Ww]ork\}",
        "method": r"\\section\*?\{[Mm]ethod(?:ology|s)?\}",
        "experiment": r"\\section\*?\{[Ee]xperiment(?:s)?\}",
    }

    # Find all section headers as boundaries
    any_section_re = re.compile(r"\\section\*?\{.*?\}")
    boundaries = [(m.start(), m.end()) for m in any_section_re.finditer(latex)]

    # Set title of all section 
    all_section_title = ['Introduction', 'Related Work', 'Methodology', 'Experiment']
    idx=0

    result: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    for key, pat in pat_map.items():
        m = re.search(pat, latex)
        if not m:
            continue
        start = m.end()
        # Find the next boundary after this header
        end = len(latex)
        for s, e in boundaries:
            if s > start:
                end = s
                break
        
        body = f'\section{{{all_section_title[idx]}}}\n\n'+latex[start:end].strip()
        idx+=1
        result[key] = (body, (start, end))

    return result


def build_task_input(section_key: SectionKey, style_guide: str) -> str:
    """Compose task input for the TEXT agent and Judge, including optional style guide and reference context."""

    guidance = (
        "You will revise a LaTeX section from an academic paper.\n"
        "Strict constraints:\n"
        "- Preserve technical correctness and any equations/macros.\n"
        "- Keep content grounded in the given text; no unverifiable claims.\n"
        "- Maintain consistent terminology with the whole paper.\n"
        "- Keep LaTeX intact (math, citations, formatting).\n"
        "- Improve clarity, structure, and concision.\n"
        "- Do not shorten excessively; keep similar length unless verbosity harms clarity.\n"
        "- If a user style guide (paper_style.md) is provided in the case folder, align tone, vocabulary, tense, and structure with it while preserving all technical content.\n"
    )

    # 全章节通用：若本章节存在图环境或图的引用，需严格保留图环境与对应引用
    guidance += (
        "  HARD REQUIREMENT (all sections): If this section contains any LaTeX figure environments or figure references,\n"
        "  preserve ALL \\begin{figure}...\\end{figure} blocks and their order.\n"
        "  Keep all \\includegraphics options/paths and \\label tokens unchanged.\n"
        "  Do NOT remove, rename, or alter any figure references (e.g., \\ref{fig:*}, \\autoref{fig:*}, \\cref{fig:*}, 'Fig.', 'Figure').\n"
        "  You MAY refine \\caption text for concision ONLY if meaning and references remain intact; do NOT convert figures into prose.\n"
    )

    # 实验部分：在保持图表环境与引用完整不变的前提下，仅允许精炼图/表注及叙述文本
    if section_key == "experiment":
        guidance += (
            "  HARD REQUIREMENT: Preserve ALL LaTeX figure environments and their order.\n"
            "  Do NOT remove or reorder any \\begin{figure}...\\end{figure} blocks.\n"
            "  Keep \\includegraphics options/paths and \\label tokens unchanged.\n"
            "  You MAY refine \\caption text for concision ONLY if meaning and references remain intact.\n"
            "  Keep figure captions concise (ideally <= 40-45 words), avoid multi-sentence narratives.\n"
            "  Do NOT convert figures into prose; only edit surrounding narrative and caption brevity.\n"
            "- HARD REQUIREMENT: Preserve ALL LaTeX table environments and their order.\n"
            "  Do NOT remove or reorder any \\begin{table}...\\end{table} or \\begin{table*}...\\end{table*} blocks.\n"
            "  Keep ALL table contents unchanged: tabular/longtable/tabularx cells, numbers, formats; keep \\label tokens unchanged.\n"
            "  Do NOT remove, rename, or alter any table references (e.g., \\ref{tab:*}, \\autoref{tab:*}, \\cref{tab:*}, 'Tab.', 'Table').\n"
            "  You MAY refine table \\caption text for concision ONLY; do NOT modify the table body content.\n"
            "  Narrative scope constraint: Only modify natural-language narrative and figure/table captions.\n"
            "  Do NOT introduce any new experiments, ablations, datasets, metrics, or results that are not present in the original text.\n"
            "  Do NOT fabricate numbers or create new figures/tables; preserve the original counts and references.\n"
        )

    # # 介绍或相关工作章节不允许删改正文的文献引用
    # if section_key == "related_work" or section_key == "introduction":

    if section_key == "introduction":
        guidance=("**It is necessary to introduce with a natural transition sentence at the end, and then list the summary of contributions (list/items of contributions), so as to clearly and structurally summarize the main contributions.**\n"+guidance)

    #     guidance=("\n**Be sure to retain all references and citations, and do not remove any of the cited literature. For example, do not delete or modify any citation commands such as \cite{key1, key2}, \citep{key}, \citeauthor{key}, etc., or their corresponding reference entries.**\n"+guidance)

        
    return (
        f"Target section: {section_key}\n\n"
        + guidance
        + ("\n\nUser-provided style guide (paper_style.md):\n" + style_guide if style_guide else "")
    )


def build_rubric(section_key: SectionKey) -> Dict[str, str]:
    """Return a single-criterion 1..5 rubric tailored to the section."""
    if section_key == "introduction":
        return {
            "criteria": (
                "Introduction quality: motivates the problem crisply, defines scope and gap,"
                " states contributions concretely (bulletable), positions vs. prior work without"
                " sweeping claims, outlines paper structure in a coherent flow, controls"
                " overall length (concise; avoid unnecessary verbosity), and avoids excessive"
                " bulleting/listing (prefer cohesive prose; if listing contributions, keep it minimal)."
            ),
            "score1_description": "Vague, off-topic, lacks problem/gap; overclaims; disorganized; unclear terminology; overly long or rambling; excessive bullet lists.",
            "score2_description": "Basic context but unclear problem/gap; contributions generic; weak positioning; flow choppy; length not well controlled (too long/verbose); too many bullets or enumerations.",
            "score3_description": "Adequate problem and contributions; some positioning; minor clarity/conciseness issues; somewhat wordy and/or list-heavy but acceptable.",
            "score4_description": "Clear problem/gap; specific contributions; balanced positioning; good readability and flow; appropriate length; minimal and focused bullets only when necessary.",
            "score5_description": "Crisp motivation; precise gap; well-scoped, verifiable contributions; nuanced positioning; excellent clarity and structure; concise prose with at most a minimal contribution list.",
        }
    if section_key == "related_work":
        return {
            "criteria": (
                "Related Work coverage: accurately summarizes major method families and key papers,"
                " compares assumptions and trade-offs to the proposed approach, cites correctly,"
                " and highlights unresolved gaps motivating this work. Additionally, NO tables are"
                " allowed in the related work section (avoid \\begin{table}/tabular), and prose should"
                " be concise (avoid verbosity; prefer compact synthesis rather than long lists)."
            ),
            "score1_description": "Missing key lines of work; inaccurate or claimy; poor citations; no comparison; contains tables and/or highly verbose lists.",
            "score2_description": "Some works mentioned but superficial; limited comparisons; spotty citations; uses tables or overly long enumerations; insufficient concision.",
            "score3_description": "Reasonable coverage; some comparative points; mostly correct; minor gaps; mostly concise; avoids tables in most parts.",
            "score4_description": "Comprehensive and accurate; clear taxonomy; fair comparisons; gaps well articulated; concise synthesis; no tables.",
            "score5_description": "Authoritative and incisive synthesis; precise attributions; comparisons clarify assumptions and trade-offs; excellent concision; strictly no tables.",
        }
    if section_key == "method":
        return {
            "criteria": (
                "Methodology quality: revolve around the core modifications and innovations; "
                "make the deltas to baselines/prior art explicit with equations or algorithmic steps; "
                "provide clear problem formalization, notation, objectives/constraints, and design rationale; "
                "state assumptions and limitations; ensure complexity and implementation details support reproducibility. "
                "Avoid peripheral/background material that belongs to other sections. "
                "Within each subsection, minimize bullet/enum lists and prefer cohesive prose; only use lists when strictly necessary and keep them short. "
                "Additionally, NO tables are allowed in the method section (avoid \\begin{table}/tabular). "
                "Keep the exposition concise and formula-centric, avoid redundancy and narrative filler."
            ),
            "score1_description": "Unclear or off-core; fails to articulate the core modifications/innovations; inconsistent notation; unverifiable claims; contains tables; excessive redundancy or peripheral background; heavy bulleting/enumeration within subsections.",
            "score2_description": "Partial details; deltas vs baseline/prior art not explicit; ambiguities in notation/rationale; includes non-core digressions or long lists; verbose and redundant; relies on multiple bullet lists instead of cohesive prose within subsections.",
            "score3_description": "Generally clear with minor gaps; some explicit focus on core modifications; equations mostly consistent; some redundancy remains; mostly avoids tables; limited bulleting with room to convert to cohesive prose.",
            "score4_description": "Well-structured and reproducible; deltas and innovations are explicit and centered; concise, formula-centric presentation; clear assumptions/limits; no tables; minimal non-core content; lists are rare, short, and justified.",
            "score5_description": "Highly rigorous and self-contained; laser-focused on core modifications and innovations with explicit deltas; precise math and algorithms; limits explicit; exemplary concision with minimal redundancy; strictly no tables; subsections use cohesive prose with at most minimal, necessary lists.",
        }
    if section_key == "experiment":
        return {
            "criteria": (
                "Experimental validity: clearly defined setup (datasets, splits, metrics, baselines),"
                " fair comparisons, ablations, diagnostics, and principled interpretation linked to the method;"
                " results grounded and reproducibility considerations addressed.\n"
                "HARD REQUIREMENT: preserve ALL LaTeX figure AND table environments and their order; do not remove or reorder any \\begin{figure}...\\end{figure} or \\begin{table}...\\end{table} (including \\begin{table*}...\\end{table*}) blocks;"
                " keep \\includegraphics options/paths, all tabular/longtable bodies, and \\label tokens unchanged.\n"
                "Figure/Table presence check: preserve the counts of figure and table environments; if either count decreases vs the original, assign a low score; if references (e.g., \\ref{fig:*}/\\ref{tab:*}, Fig./Figure/Tab./Table) exist but corresponding environments are missing, assign a low score.\n"
                "Caption specificity and relevance: captions must thoroughly describe the figure/table with content-critical information.\n"
                " For figures: what is plotted; the meaning and units of axes; metric definitions; dataset/split/conditions; legend/marker/line semantics when encoding categories; and the key takeaway.\n"
                " For tables: what each column/row means; metric definitions; dataset/split/conditions; the key comparison or takeaway. Prefer precise, compact wording (1–2 sentences acceptable). Avoid styling-only commentary (e.g., 'the plot title is ...', 'gridlines', color adjectives without category mapping). Focus on interpretability, not decoration.\n"
                "Scope restriction: Only modify narrative text and figure/table captions. Do NOT introduce new experiments, ablations, datasets, metrics, or fabricated results beyond the original."
            ),
            "score1_description": "Fails hard requirement (any figure/table removed/reordered or includegraphics/tabular/label altered) or setup unclear; unsupported claims; fabricates new experiments/results; captions dominated by styling commentary and missing content-critical elements; figure/table count decreased vs original or references present without corresponding environments.",
            "score2_description": "Basic setup but incomplete; weak comparisons; preserves floats but with minor structural edits; captions partially describe content but omit multiple key elements (e.g., axes/units, metric definition, legend/column meaning, or takeaway) and/or include noticeable styling-only fluff.",
            "score3_description": "Reasonable setup and comparisons; preserves figure/table environments (only trivial whitespace differences); captions generally informative with minor omissions (e.g., units or brief metric definition) and limited, non-distracting styling commentary.",
            "score4_description": "Thorough setup with strong baselines and diagnostics; strictly preserves figure/table environments (counts and order); captions consistently content-focused and thorough: state what is shown, axes/units or column meanings, metric definitions, dataset/splits/conditions as relevant, plus a clear takeaway; no redundant styling commentary.",
            "score5_description": "Exemplary rigor: comprehensive setup, ablations, uncertainty/robustness; impeccable preservation of figure/table code blocks (counts/order) and labels; captions crisp, complete, and interpretation-centric across figures and tables, covering content, axes/units or column meanings, metric definitions, legend semantics, dataset/splits/conditions (and sample sizes/time windows when relevant), with explicit takeaways; strictly avoids styling/formatting narration.",
        }
    # Default fallback rubric
    return {
        "criteria": "Overall writing quality and alignment with paper context and references.",
        "score1_description": "Poor quality; unclear; off-context.",
        "score2_description": "Limited clarity; important gaps.",
        "score3_description": "Adequate but improvable.",
        "score4_description": "Good quality; minor issues.",
        "score5_description": "Excellent; clear, rigorous, and well-structured.",
    }


# Heuristics for experiment figure/table presence
_FIG_BEGIN_RE = re.compile(r"\\begin\{figure\*?\}", re.IGNORECASE)
_FIG_REF_PATTERNS = [
    re.compile(r"\\ref\{fig:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\autoref\{fig:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\cref\{fig:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\Cref\{fig:[^}]+\}", re.IGNORECASE),
    re.compile(r"\bFig\.?\s*\d+", re.IGNORECASE),
    re.compile(r"\bFigure\s*\d+", re.IGNORECASE),
]
_TABLE_BEGIN_RE = re.compile(r"\\begin\{table\*?\}", re.IGNORECASE)
_TABLE_REF_PATTERNS = [
    re.compile(r"\\ref\{tab:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\autoref\{tab:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\cref\{tab:[^}]+\}", re.IGNORECASE),
    re.compile(r"\\Cref\{tab:[^}]+\}", re.IGNORECASE),
    re.compile(r"\bTab\.?\s*\d+", re.IGNORECASE),
    re.compile(r"\bTable\s*\d+", re.IGNORECASE),
]


def _count_figure_envs(tex: str) -> int:
    return len(_FIG_BEGIN_RE.findall(tex or ""))

def _count_table_envs(tex: str) -> int:
    return len(_TABLE_BEGIN_RE.findall(tex or ""))


def _has_figure_refs(tex: str) -> bool:
    t = tex or ""
    for pat in _FIG_REF_PATTERNS:
        if pat.search(t):
            return True
    return False

def _has_table_refs(tex: str) -> bool:
    t = tex or ""
    for pat in _TABLE_REF_PATTERNS:
        if pat.search(t):
            return True
    return False


def evolve_section(
    *,
    section_key: SectionKey,
    original_text: str,
    task_input: str,
    rubric: Dict[str, str],
    model_name: Optional[str],
    judge_model: Optional[str],
    iters: int,
    population: int = 1,
) -> Tuple[str, float]:
    """Run Judge-guided iterative modification and return best text and score."""
    judge = TextEvalAgent(model_name=judge_model)
    modifier = TextModifierAgent(model_name=model_name)

    # Evaluate original
    best_text = original_text
    ev = judge.evaluate(original_text, rubric, task_input)
    best_score = float(ev.get("mean_score", 0.0))
    suggestions = ev.get("suggestions", [])

    # For experiment: track original figure/table count
    orig_fig_count = _count_figure_envs(original_text) if section_key == "experiment" else 0
    orig_tab_count = _count_table_envs(original_text) if section_key == "experiment" else 0

    pop = max(1, int(population or 1))
    for _ in range(max(0, iters)):
        # 生成多个候选个体，并从中选择评分最高者作为下一代父本
        top_text = None
        top_score = -1.0
        top_suggestions: List[str] = []

        for _k in range(pop):
            mod = modifier.modify(
                current_text=best_text,
                task_input=task_input,
                rubric=rubric,
                judge_suggestions=suggestions,
                temperature=1,
                use_judge_suggestions=True,
                section_key=section_key
            )
            cand_text = mod.get("text", best_text)
            ev2 = judge.evaluate(cand_text, rubric, task_input)
            score2 = float(ev2.get("mean_score", 0.0))

            # 实验章节：若候选的 figure/table 数量少于原始，则直接将分数降为 1.0，并补充提示
            suggestions2 = ev2.get("suggestions", [])
            if section_key == "experiment":
                cand_fig_count = _count_figure_envs(cand_text)
                cand_tab_count = _count_table_envs(cand_text)
                if (orig_fig_count > cand_fig_count) or (orig_tab_count > cand_tab_count):
                    score2 = 1.0
                    suggestions2 = (suggestions2 or []) + [
                        "Restore all original \\begin{figure}...\\end{figure} and \\begin{table}...\\end{table} environments to match the original counts and order; keep \\includegraphics paths/options, tabular bodies, and \\label intact; only refine captions."
                    ]

            if score2 > top_score:
                top_score = score2
                top_text = cand_text
                top_suggestions = suggestions2

        # 与当前最优比较，若提升则更新；否则刷新建议以帮助跳出局部最优
        if top_text is not None and top_score >= best_score:
            best_text = top_text
            best_score = top_score
            suggestions = top_suggestions
        else:
            suggestions = top_suggestions or suggestions

    return best_text, best_score


def evolve_paper(latex_content: str, config: dict, style_guide: str, sections: str='all'):

    split_sections = extract_sections(latex_content)
   
    # Process in specified order
    order: List[Tuple[SectionKey, str]] = [
        ("introduction", "introduction.tex"),
        ("related_work", "related_work.tex"),
        ("method", "method.tex"),
        ("experiment", "experiment.tex"),
    ]

    # Normalize user-specified sections

    selected: Optional[List[str]] = None
    if sections != "all":
        aliases = {
            "intro": "introduction",
            "introduction": "introduction",
            "related": "related_work",
            "related_work": "related_work",
            "relatedwork": "related_work",
            "rw": "related_work",
            "method": "method",
            "methods": "method",
            "methodology": "method",
            "exp": "experiment",
            "experiment": "experiment",
            "experiments": "experiment",
        }
        selected = []
        for token in sections.split(","):
            key = aliases.get(token.strip(), None)
            if key and key not in selected:
                selected.append(key)
        if not selected:
            raise RuntimeError(f"[WARN] No valid sections parsed from --sections='{sections}'. Nothing to do.")
        
    else:
        selected=["introduction", "related_work", "method", "experiment"]

    all_paper_content=''
    for key, out_name in order:
        if selected is not None and key not in selected:
            continue
        if key not in split_sections:
            raise RuntimeError(f"[WARN] Section not found in LaTeX: {key}")
        
        original_body, _span = split_sections[key]
        task_input = build_task_input(key, style_guide)
        rubric = build_rubric(key)

        logging.info(f"[EVO] Evolving section: {key} -> {out_name}")
        best, score = evolve_section(
            section_key=key,
            original_text=original_body,
            task_input=task_input,
            rubric=rubric,
            model_name=config['write_model_name'],
            judge_model=config['judge_model'],
            iters=config['evol_iter'],
            population=config['population'],
        )

        all_paper_content+=(best.strip()+'\n')

        logging.info(f"mean_score={score:.2f}")
    return all_paper_content
