import os
import re
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion  # type: ignore


DocLike = Dict[str, Any]  # {"page_content": str, "metadata": {"Food code": str, ...}}


class DietAI24:
    """
    DietAI24 for TEXT input.
    Workflow unchanged conceptually:
      1) Normalize the meal description
      2) Generate query variants & retrieve candidates
      3) Select food codes conservatively from retrieved context

    Vector DB contract:
      - Pass `vectordb` as a callable: (query: str, k: int) -> List[DocLike]
        or an object exposing `.similarity_search(query, k=...)` or `.search(query, top_k=...)`.
      - Each DocLike must include metadata["Food code"].

    Notes:
      - Uses OpenAI for small reasoning steps.
      - Robust JSON parsing & light retry logic.
    """

    def __init__(
        self,
        model_name: str,
        vectordb: Optional[Union[Callable[[str, int], List[DocLike]], Any]] = None,
        *,
        temperature: float = 0.0,
        top_k_per_item: int = 8,
        request_timeout: Optional[float] = None,
        max_retries: int = 3,
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

        self.model = model_name
        self.temperature = temperature
        self.top_k_per_item = top_k_per_item
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self.vectordb = vectordb

        # Conversation history (kept minimal)
        system_prompt = (
            "You are an American food assistant. You identify foods, assign appropriate food codes, and estimate portion weights from context. "
            "You are conservative and do not invent codes that are not in the provided list. "
            "When unsure, you return null rather than guessing."
        )
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # -------------------------
    # Low-level OpenAI chat with retries
    # -------------------------

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 300,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatCompletion:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=self.request_timeout,
                    response_format=response_format,
                )
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0

    # -------------------------
    # Utility: JSON parsing & code extraction
    # -------------------------

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str):
            return None
        txt = text.strip()
        # Strip code fences
        if txt.startswith("```"):
            nl = txt.find("\n")
            if nl != -1:
                txt = txt[nl + 1 :].strip()
            if txt.endswith("```"):
                txt = txt[:-3].strip()

        # Find first {...} block
        stack, start = [], None
        for i, ch in enumerate(txt):
            if ch == "{":
                if start is None:
                    start = i
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        cand = txt[start : i + 1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            start = None
        try:
            return json.loads(txt)
        except Exception:
            return None

    @staticmethod
    def _safe_find_8digit_codes(text: str) -> List[str]:
        return re.findall(r"\b\d{8}\b", text or "")

    # -------------------------
    # Vector search adapter
    # -------------------------

    def _search_vector(self, query: str, k: Optional[int] = None) -> List[DocLike]:
        k = k or self.top_k_per_item
        if self.vectordb is None:
            return []
        if callable(self.vectordb):
            return [self._coerce_doc(d) for d in self.vectordb(query, k)]
        if hasattr(self.vectordb, "similarity_search"):
            return [self._coerce_doc(d) for d in self.vectordb.similarity_search(query, k=k)]
        if hasattr(self.vectordb, "search"):
            return [self._coerce_doc(d) for d in self.vectordb.search(query, top_k=k)]
        raise NotImplementedError("vectordb must be callable or have similarity_search/search methods")

    @staticmethod
    def _coerce_doc(d: Any) -> DocLike:
        if isinstance(d, dict):
            return {"page_content": str(d.get("page_content", "")), "metadata": dict(d.get("metadata", {}) or {})}
        page = getattr(d, "page_content", "")
        meta = getattr(d, "metadata", {}) or {}
        return {"page_content": str(page), "metadata": dict(meta)}

    # -------------------------
    # STEP 1: Normalize/structure the meal description (TEXT)
    # -------------------------

    def _normalize_meal_text(self, meal_text: str) -> Dict[str, Any]:
        """
        Parse a free-text meal description into a normalized description and item list.
        Returns:
        {
          "normalized": "Short canonical description of the meal ...",
          "items": [{"label": "pepperoni pizza, medium crust", "details": "..."}, ...]
        }
        """
        instr = (
            "You will read a short, free-text meal description and extract distinct food/drink items. "
            "Return JSON with two fields:\n"
            '  - "normalized": one concise sentence describing the meal in neutral terms\n'
            '  - "items": an array of objects with "label" (canonical short name) and optional "details"\n\n'
            "Rules:\n"
            "- Combine obvious descriptors (e.g., 'medium crust pepperoni pizza').\n"
            "- Keep labels short and database-friendly.\n"
            "- Do not include quantities unless clearly stated.\n"
            "- If nothing edible is present, return items = []."
        )
        messages = [
            self.messages[0],
            {"role": "user", "content": f"{instr}\n\nMeal:\n{meal_text}"},
        ]
        try:
            resp = self._chat(messages, max_tokens=220, response_format={"type": "json_object"})
            txt = resp.choices[0].message.content or ""
            data = json.loads(txt)
        except Exception:
            resp = self._chat(messages, max_tokens=220)
            txt = resp.choices[0].message.content or ""
            data = self._extract_json_obj(txt) or {"normalized": meal_text.strip(), "items": []}

        # Normalize shape
        norm = {
            "normalized": str(data.get("normalized", meal_text)).strip(),
            "items": [],
            "raw": txt,
        }
        items = data.get("items", [])
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("label"):
                    norm["items"].append({"label": str(it["label"]).strip(), "details": str(it.get("details", "")).strip()})
        return norm

    # -------------------------
    # STEP 2: Generate query variants for retrieval (TEXT)
    # -------------------------

    def _generate_query_variants(self, short_label: str) -> List[str]:
        """
        Produce 4 short variants suitable for retrieval (JSON list).
        """
        instr = (
            "Generate four differently worded search queries for a food database entry. "
            "Vary: main ingredient, preparation, crust/type/cut/brand (if applicable), dietary attributes, cuisine, serving style. "
            "Return ONLY a JSON array of strings."
        )
        messages = [
            {"role": "system", "content": "You generate concise retrieval queries."},
            {"role": "user", "content": f"{instr}\n\nItem label:\n{short_label}"},
        ]
        out: List[str] = []
        try:
            resp = self._chat(messages, max_tokens=160, response_format={"type": "json_object"})
            txt = resp.choices[0].message.content or "[]"
            data = json.loads(txt)
            if isinstance(data, list):
                out = [str(x) for x in data if isinstance(x, str)]
            elif isinstance(data, dict) and isinstance(data.get("queries"), list):
                out = [str(x) for x in data["queries"] if isinstance(x, str)]
        except Exception:
            resp = self._chat(messages, max_tokens=160)
            txt = resp.choices[0].message.content or ""
            data = self._extract_json_obj(txt)
            if isinstance(data, list):
                out = [str(x) for x in data if isinstance(x, str)]
            elif isinstance(data, dict) and isinstance(data.get("queries"), list):
                out = [str(x) for x in data["queries"] if isinstance(x, str)]

        # Fallback: simple variants if LLM parsing failed
        if not out:
            base = short_label.strip()
            out = [
                base,
                f"{base} typical preparation",
                f"{base} restaurant style",
                f"{base} packaged brand",
                f"{base} US style",
            ]
        return out[:4]

    # -------------------------
    # STEP 3: Select codes from retrieved context (TEXT-only)
    # -------------------------

    def _select_codes_from_context(self, meal_text: str, context_text: str) -> Dict[str, Any]:
        """
        Ask model to select only codes present in context that cover all distinct foods in the meal.
        Returns {"food_codes": [...], "assignments": [{"item":"...","code":"..."}]}
        """
        instr = (
            "Given a meal description and candidate food codes with short descriptions, select codes that cover all "
            "distinct foods. Use ONLY codes present in the context. If no suitable code is present for an item, omit it.\n\n"
            "Return ONLY JSON with this schema:\n"
            '{ "food_codes": ["12345678", "..."], "assignments": [{"item":"<label>","code":"12345678"}] }'
        )
        user_text = (
            f"{instr}\n\nMeal description:\n{meal_text}\n\n"
            f"Context (candidate codes):\n{context_text}"
        )
        messages = [
            {"role": "system", "content": "You are conservative and never invent codes outside the provided context."},
            {"role": "user", "content": user_text},
        ]
        try:
            resp = self._chat(messages, max_tokens=240, response_format={"type": "json_object"})
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
        except Exception:
            resp = self._chat(messages, max_tokens=240)
            raw = resp.choices[0].message.content or ""
            data = self._extract_json_obj(raw) or {}

        # Normalize
        codes = data.get("food_codes")
        if not isinstance(codes, list):
            codes = self._safe_find_8digit_codes(raw)
        codes = [c for c in codes if isinstance(c, str)]

        assignments = []
        if isinstance(data.get("assignments"), list):
            for a in data["assignments"]:
                if isinstance(a, dict) and isinstance(a.get("item"), str) and isinstance(a.get("code"), str):
                    assignments.append({"item": a["item"], "code": a["code"]})

        return {"food_codes": codes, "assignments": assignments, "raw": raw}

    # -------------------------
    # PUBLIC: Recognize food codes from TEXT
    # -------------------------

    def recognize_food_from_text(self, meal_text: str) -> Dict[str, Any]:
        """
        Main entry for text input.
        Returns:
        {
          "ok": bool,
          "food_codes": List[str] | None,
          "normalized_description": str,
          "items": List[{"label":..., "details":...}],
          "context_text": str,
          "assignments": List[{"item":..., "code":...}],
          "raw_selection": str,
          "error": str | None
        }
        """
        # 1) Normalize/structure
        norm = self._normalize_meal_text(meal_text)
        items = norm["items"]

        # 2) Retrieval: multi-query per item
        all_docs: List[DocLike] = []
        for it in items or [{"label": norm["normalized"]}]:
            label = it["label"]
            queries = self._generate_query_variants(label)
            # Also include the raw label as an anchor
            if label not in queries:
                queries.append(label)
            for q in queries:
                all_docs.extend(self._search_vector(q, k=self.top_k_per_item))

        # Deduplicate by Food code
        unique: Dict[str, DocLike] = {}
        for d in all_docs:
            code = (d.get("metadata") or {}).get("Food code")
            if code and code not in unique:
                unique[code] = d
        deduped = list(unique.values())

        # Compact context for the selector
        lines = []
        for d in deduped:
            code = (d.get("metadata") or {}).get("Food code", "")
            desc = (d.get("page_content") or "").strip().replace("\n", " ")
            if code and desc:
                lines.append(f"Food code: {code} | {desc}")
        context_text = "\n".join(lines)

        # 3) Selection
        sel = self._select_codes_from_context(meal_text, context_text)
        codes = sel["food_codes"]
        ok = len(codes) > 0

        # Minimal conversation log (optional)
        self.messages.append({"role": "user", "content": meal_text})
        self.messages.append({"role": "assistant", "content": json.dumps({"normalized": norm["normalized"], "items": items})})
        self.messages.append({"role": "assistant", "content": sel["raw"]})

        return {
            "ok": ok,
            "food_codes": codes if ok else None,
            "normalized_description": norm["normalized"],
            "items": items,
            "context_text": context_text,
            "assignments": sel["assignments"],
            "raw_selection": sel["raw"],
            "error": None if ok else "no_codes_found",
        }

    # -------------------------
    # Portion reference helper (unchanged)
    # -------------------------

    def portion_reference_to_text(
        self,
        df_ref: pd.DataFrame,
        show_header: bool = True,
        weight_unit: str = "g",
    ) -> str:
        if df_ref is None or df_ref.empty:
            return "No reference information available."
        required_cols = {"Main food description", "Portion description", "Portion weight (g)"}
        if not required_cols.issubset(df_ref.columns):
            missing = sorted(required_cols - set(df_ref.columns))
            return f"No reference information available. Missing columns: {', '.join(missing)}"

        lines: List[str] = []
        for food, group in df_ref.groupby("Main food description"):
            if show_header:
                lines.append(f"Reference information for {food}:")
            for _, row in group.iterrows():
                portion = str(row.get("Portion description", "")).strip()
                weight = row.get("Portion weight (g)", "")
                if portion and pd.notnull(weight):
                    lines.append(f"{portion} = {weight}{weight_unit}")
        return "\n".join(lines) if lines else "No reference information available."

    # -------------------------
    # Weight estimation (TEXT-first; image optional)
    # -------------------------

    def _build_weight_prompt_text(
        self,
        meal_text: str,
        target_item: str,
        scenario: str,
        portion_reference_text: Optional[str],
        use_weight_reference: bool,
    ) -> str:
        """
        Build a weight-estimation prompt that *must* ground in the original meal description.
        """
        assert scenario in {"before", "after"}
        head = (
            "You are estimating the *weight in grams* for a specific item mentioned in a meal description. "
            "Use only cues from the meal text (quantities, packaging, size adjectives, typical units). "
            "If text is insufficient, return null rather than guessing.\n\n"
        )
        if scenario == "after":
            head += (
                "If the wording suggests the item was finished or only traces remain, return 0.\n"
            )

        rules = (
            "Rules:\n"
            "- Use only the meal text; do not invent container sizes or portion counts that are not implied.\n"
            "- If the text gives a clear count (e.g., 'one slice', 'a carton'), translate to grams using typical servings.\n"
            "- If the text gives only qualitative size (e.g., 'medium crust pizza'), make a conservative assumption.\n"
            "- If still ambiguous, set weight_grams to null.\n"
        )

        ref = ""
        if use_weight_reference and portion_reference_text:
            ref = (
                "\nOptional portion reference (use only if it matches the item semantics):\n"
                f"{portion_reference_text}\n"
            )

        schema = (
            '\nReturn ONLY JSON (no backticks) with this schema:\n'
            '{ "weight_grams": number | null, "reasoning": "one short sentence citing meal-text cues" }'
        )

        return (
            f"{head}"
            f"Meal text:\n{meal_text}\n\n"
            f"Target item:\n{target_item}\n\n"
            f"{rules}"
            f"{ref}"
            f"{schema}"
        )

    def _estimate_weight_text_only(
        self,
        meal_text: str,
        target_item: str,
        scenario: str,
        portion_reference_text: Optional[str],
        use_weight_reference: bool,
    ) -> Dict[str, Any]:
        prompt = self._build_weight_prompt_text(
            meal_text=meal_text,
            target_item=target_item,
            scenario=scenario,
            portion_reference_text=portion_reference_text,
            use_weight_reference=use_weight_reference,
        )
        messages = [self.messages[0], {"role": "user", "content": prompt}]

        try:
            resp = self._chat(messages, max_tokens=240, response_format={"type": "json_object"})
            txt = resp.choices[0].message.content or ""
            data = json.loads(txt)
        except Exception:
            resp = self._chat(messages, max_tokens=240)
            txt = resp.choices[0].message.content or ""
            data = self._extract_json_obj(txt) or {"weight_grams": None, "reasoning": txt}

        out = {
            "weight_grams": data.get("weight_grams", None),
            "reasoning": data.get("reasoning", ""),
            "raw": txt,
        }
        if out["weight_grams"] is not None:
            try:
                out["weight_grams"] = float(out["weight_grams"])
            except Exception:
                out["weight_grams"] = None
        return out

    def estimate_weight_before_eating(
        self,
        meal_text: str,
        assumed_food: Optional[str] = None,
        portion_reference_text: Optional[str] = None,
        use_weight_reference: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate weight for a *specific* item grounded in the original meal text.
        If `assumed_food` is None and the meal contains exactly one parsed item, use that item.
        """
        if not isinstance(meal_text, str) or not meal_text.strip():
            raise ValueError("meal_text is required and cannot be empty.")
        
        target = assumed_food
        if target is None:
            norm = self._normalize_meal_text(meal_text)
            if len(norm["items"]) == 1:
                target = norm["items"][0]["label"]
            else:
                raise ValueError("assumed_food is required when the meal mentions multiple items.")

        return self._estimate_weight_text_only(
            meal_text=meal_text,
            target_item=target,
            scenario="before",
            portion_reference_text=portion_reference_text,
            use_weight_reference=use_weight_reference,
        )

    def estimate_weight_after_eating(
        self,
        meal_text: str,
        assumed_food: Optional[str] = None,
        portion_reference_text: Optional[str] = None,
        use_weight_reference: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate *remaining* weight for a specific item grounded in the original meal text.
        """
        if not isinstance(meal_text, str) or not meal_text.strip():
            raise ValueError("meal_text is required and cannot be empty.")
        
        target = assumed_food
        if target is None:
            norm = self._normalize_meal_text(meal_text)
            if len(norm["items"]) == 1:
                target = norm["items"][0]["label"]
            else:
                raise ValueError("assumed_food is required when the meal mentions multiple items.")

        return self._estimate_weight_text_only(
            meal_text=meal_text,
            target_item=target,
            scenario="after",
            portion_reference_text=portion_reference_text,
            use_weight_reference=use_weight_reference,
        )

