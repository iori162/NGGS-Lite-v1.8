# NGGS-Lite v1.8: Full Refactoring, Implementation, and Optimization - Statement of Work (for AI Agent Jules)

**Document Version:** 3.1 (Optimized for AI Agent Jules)
**Project:** NGGS-Lite v1.8 Full Implementation and Refinement
**Target Script:** `NGGS-Lite_v1_8_refactored.py` (The 18-part consolidated Python script)
**Primary AI Agent:** Jules
**Date:** May 27, 2025

## 1. 🎯 Overall Project Goal

### 1.1. Primary Objective
Your primary objective, Jules, is to transform the provided `NGGS-Lite_v1_8_refactored.py` script into a **fully executable, robust, and feature-complete Python application**. This requires:
1.  Rectifying all identified syntactic errors, logical flaws, and type inconsistencies.
2.  Implementing all stubbed methods and missing functionalities as detailed in the "NGGS-Lite v1.8 (Gemini Edition) User Guide" (hereafter "User Guide"). [cite: 535]
3.  Addressing all issues and completing all unimplemented items specified in the "NGGS-Lite v1.8 (Refactoring Edition) Final Check and Commentary" document (hereafter "Final Check Document"). [cite: 553]
4.  Optimizing the codebase for performance, maintainability, and adherence to Python best practices.

### 1.2. Key Deliverables & Success Criteria
The ultimate deliverable is a single Python script named `NGGS-Lite_v1.8_Jules_Implemented.py`. This script must meet the following success criteria:

* **Full Executability:** The script must execute without any Python errors (syntax, indentation, name, type, runtime, etc.) when run with Python 3.x. [cite: 536] This will be verified using `python -m py_compile NGGS-Lite_v1.8_Jules_Implemented.py` and `mypy NGGS-Lite_v1.8_Jules_Implemented.py` (with appropriate strictness settings).
* **Complete Feature Implementation (User Guide Alignment):** All functionalities described in the User Guide [cite: 537] must be fully implemented and operational. This includes, but is not limited to:
    * ETI (Existential Tremor Index) and RI (Readability Index) evaluations, including all their specified components. [cite: 538]
    * The Phase Transition Model and Four-Layer Structure analysis and their application in text generation/evaluation. [cite: 539]
    * Emotion Arc score calculation and Colloquial/Gothic Blend score calculation. [cite: 539]
    * NDGS (Neo Dialogue Generation System) input parsing as per User Guide section 7.1. [cite: 540]
    * GLCAI (Gothic Lexicon Curator AI) vocabulary feedback data generation and output as per User Guide section 7.2. [cite: 540]
    * Full implementation of all `TextProcessor._create_*_template` methods as outlined in Part 14 of this SoW. [cite: 541]
    * A fully functional `TextProcessor.process` method, incorporating a robust iterative improvement loop as described in Part 13. [cite: 541]
    * Generation of functional HTML reports for individual jobs and batch summaries, with report paths correctly stored in the results dictionary[cite: 542], as specified in Part 14 and Part 16.
* **Adherence to Design Principles:**
    * Emulate the modular structure, robust error handling (using the `Result` type and custom `NGGSError` subclasses), and data validation practices found in the reference `dialogue_generator.py` (NDGS v4.9α) script. [cite: 543]
    * Strictly follow the "Stand-alone Priority" and "Quality Concentration Strategy" as detailed in the "System Development Roadmap (v2.0)" (hereafter "Roadmap"). [cite: 544]
* **Resolution of "Final Check Document" Items:** All points, unimplemented items, and suggested improvements from the "Final Check Document" [cite: 553] must be comprehensively addressed and implemented. This is a critical requirement.
* **Code Quality Standards:**
    * **PEP8 Compliance:** The entire codebase must strictly adhere to PEP8 standards.
    * **Type Hinting:** Implement comprehensive type hints for all function and method signatures, and for all significant variables.
    * **Docstrings:** Provide clear, concise, and informative docstrings (Google style) for all classes, methods, and functions.
    * **Comments:** Include comments in English or Japanese to clarify complex logic sections or important design decisions.

## 2. 🧾 Context and Provided Materials

### 2.1. Essential Reference Documents for Your Task, Jules:
1.  **User Guide:** `NGGS-Lite-v1.8-Gemini-Edition-user-guide.txt` (Details all features and functionalities).
2.  **Roadmap:** `System_Development_New_Roadmap_v2.0.txt` (Provides strategic development context).
3.  **NDGS Script (Reference):** `dialogue_generator.py` (NDGS v4.9α) (For design patterns and structural best practices).
4.  **Target Script (Your starting point):** `NGGS-Lite_v1.8_refactored.py` (The 18-part consolidated script).
5.  **Final Check Document (Crucial To-Do List):** "NGGS-Lite v1.8 (Refactoring Edition) Final Check and Commentary" (This document, though not directly provided to me, is referenced heavily; you must ensure all its points are actioned as per the original SoW's intent). [cite: 553]

### 2.2. Current State of the Target Script (Summary from SoW based on Final Check Document)
* **Implemented Base:** Core class skeletons (NGGSConfig, TextProcessor, LLMClient, Evaluators, etc.), basic error handling, logging, utilities, template loading, and vocabulary parsing. [cite: 554]
* **Critical Unimplemented/To-Do Areas (Your primary focus):**
    * All `TextProcessor._create_*_template` methods (Part 14). [cite: 555]
    * `TextProcessor.process` improvement loop logic (Part 13). [cite: 556]
    * `TextProcessor._generate_text` for initial and improvement modes (Part 14). [cite: 557]
    * Activation of `GLCAI` feedback calls (`track_vocabulary_usage`) within `TextProcessor.process`. [cite: 558]
    * Full NDGS parameter mapping in `NDGSIntegration.parse` (Part 15). [cite: 559]
    * Addition of specified fields to `NGGSConfig` (Part 1). [cite: 560]
    * Implementation of `_calculate_emotion_arc_score` and `_calculate_colloquial_score` (Part 11). [cite: 561]
    * Refinement of ETI/RI heuristics (Part 9) and Phase Detection (Part 10) per User Guide/Roadmap. [cite: 562]
    * HTML report generation in `TextProcessor._finalize_results` (Part 14). [cite: 563]
    * Consolidation of all import statements into Part 1. [cite: 563]

### 2.3. Execution Environment
* **Python:** Version 3.x (ensure compatibility with dataclasses, pathlib, typing).
* **Primary LLM:** Google Gemini API (via `google-generativeai` library).
* **Optional Dependencies:** `json_repair`, `concurrent.futures`. Implement conditional usage for these.

## 3. 📝 Detailed Requirements (Per Part of the Target Script)

### 3.1. General Directives for Your Implementation Process, Jules:
* **Action "Final Check Document":** This is paramount. Every point, unimplemented item, and suggested improvement from the "Final Check Document" [cite: 553] must be meticulously addressed.
* **Emulate NDGS Design Patterns:** Where applicable, adopt the robust modular design, exception handling, data validation, and clear process flow demonstrated in the `dialogue_generator.py` (NDGS Script)[cite: 567].
* **Zero-Error Mandate:** The final script must pass `python -m py_compile NGGS-Lite_v1.8_Jules_Implemented.py` and `mypy NGGS-Lite_v1.8_Jules_Implemented.py` without errors. [cite: 569]
* **User Guide Adherence:** All implemented functionalities must align precisely with the User Guide. [cite: 570] If ambiguities arise, prioritize the User Guide.
* **Roadmap Alignment:** Ensure your implementation reflects the "Quality Concentration Strategy" from the Roadmap. [cite: 571]
* **Documentation:** Any significant deviations, assumptions, or complex decisions made during implementation must be documented as comments in the code or in your submission notes.

### 3.2. Specific Part-by-Part Implementation & Refinement Instructions:

**Part 1: Imports, Constants, Core Configuration (NGGSConfig)**
* **Objective:** Finalize `NGGSConfig` with all necessary parameters and defaults; consolidate all script imports.
* **Tasks:**
    1.  **Add to `NGGSConfig` (ensure these fields are present and correctly defaulted as per SoW v3.0):**
        * `GEMINI_SAFETY_SETTINGS: Dict[str, str]` (Default to `BLOCK_NONE` for all harm categories, or align with Gemini API best practices for creative text).
        * `LOG_MAX_BYTES_DEFAULT_VAL: int` (e.g., `5 * 1024 * 1024`).
        * `LOG_BACKUP_COUNT_DEFAULT_VAL: int` (e.g., `3`).
        * `BATCH_MAX_WORKERS_DEFAULT: int` (e.g., `max(1, os.cpu_count() // 2 if os.cpu_count() else 1)`).
        * `BATCH_FILE_PATTERN_DEFAULT: str` (e.g., `"*.txt"`).
        * `PERSPECTIVE_MODE_DEFAULT: str` (e.g., `"subjective_first_person"`).
        * `PHASE_FOCUS_DEFAULT: str` (e.g., `"balanced"`).
        * `COLLOQUIAL_LEVEL_DEFAULT: str` (e.g., `"medium"`).
        * `NARRATIVE_THEME_DEFAULT: str` (e.g., `"記憶回帰型"`).
        * `ENABLE_NDGS_PARSER_DEFAULT: bool` (e.g., `True`).
        * `ENABLE_GLCAI_FEEDBACK_DEFAULT: bool` (e.g., `True`).
        * `ENABLE_JSON_REPAIR_DEFAULT: bool` (e.g., `False`).
    2.  **Consolidate All Imports:** Move all `import` statements to the beginning of Part 1. Organize them: 1. Python standard library, 2. Third-party libraries, 3. Project-specific (none for a single file). Remove duplicates and unused imports. Optimize `typing` imports (e.g., use `TypeAlias`). [cite: 578]
    3.  **Define `CONCURRENT_FUTURES_AVAILABLE: Final[bool]`:** Set this flag based on a `try-except ImportError` for `concurrent.futures`. [cite: 579]
    4.  Remove the dummy `ConfigurationError` from Part 1 (it's formally defined in Part 2). [cite: 580]
* **Priority:** High.

**Part 2: Core Exceptions and Result Type**
* **Objective:** Solidify the custom error handling framework for clarity and robustness.
* **Tasks:**
    1.  **Enrich `NGGSError.details`:** When instantiating `NGGSError` or its subclasses, systematically populate the `details` dictionary with relevant context (e.g., `failed_component`, `operation`, `input_preview`, `original_exception_type`, `original_exception_message`). [cite: 583]
    2.  **Populate `LLMResponse.metadata`:** Modify `LLMClient._call_gemini` (Part 5) to fully populate the `metadata` field of the `LLMResponse` object. This should include API call duration, token usage (prompt, candidates, total, from `response.usage_metadata`), and any other relevant API-returned metadata. [cite: 584]
* **Priority:** Medium.

**Part 3: Utility Functions (Logging, Text, File I/O)**
* **Objective:** Ensure all utility functions are robust, configurable, and correctly implemented.
* **Tasks:**
    1.  **Enhance `setup_logging`:** Ensure it uses `LOG_MAX_BYTES_DEFAULT_VAL` and `LOG_BACKUP_COUNT_DEFAULT_VAL` from the `NGGSConfig` instance for `RotatingFileHandler` parameters. [cite: 588]
    2.  **Expand `get_metric_display_name`:** Review its `display_map`. Ensure it covers all keys from the finalized `DEFAULT_EVALUATION_TEMPLATE` (Part 4) and all relevant `NGGSConfig` parameter keys that might be displayed. [cite: 589]
    3.  **Implement `Enum` Serialization in `CompactJSONEncoder`:** Add logic to its `default()` method to serialize `Enum` objects using their `value` attribute (e.g., `if isinstance(obj, Enum): return obj.value`). [cite: 590]
* **Priority:** Medium.

**Part 4: Utility Functions (Continued), LLM Client (Initialization)**
* **Objective:** Finalize template management and ensure correct LLM client (Gemini) initialization.
* **Tasks:**
    1.  **Implement Safety Settings in `LLMClient._initialize_gemini`:** Ensure this method correctly retrieves `GEMINI_SAFETY_SETTINGS` from `self.config` and passes them to `genai.GenerativeModel(..., safety_settings=...)`. [cite: 593]
    2.  **Finalize Default Template Content:** Critically review and finalize the content of `DEFAULT_GENERATION_TEMPLATE`, `DEFAULT_EVALUATION_TEMPLATE`, and `DEFAULT_IMPROVEMENT_TEMPLATE`. The `DEFAULT_EVALUATION_TEMPLATE`'s JSON output structure must explicitly list all evaluation criteria detailed in User Guide Section 1.2 (including all ETI components, RI components, etc., intended for LLM scoring). [cite: 594, 595]
    3.  **Verify `load_template` Call:** Confirm that `load_template` is called from `setup_components` (Part 17) with the `config` instance, ensuring it prioritizes file-based templates using `config.DEFAULT_TEMPLATES_DIR` before falling back to `DEFAULT_TEMPLATES`. [cite: 596]
* **Priority:** High.

**Part 5: LLM Client (Generation Logic)**
* **Objective:** Create a highly resilient LLM interaction layer with precise error handling and response parsing.
* **Tasks:**
    1.  **Update `_setup_mock` for Full Coverage:** The mock JSON evaluation response must include all keys from the finalized `DEFAULT_EVALUATION_TEMPLATE` (Part 4). Mock scores should be realistic (e.g., 1.0-5.0), and reasons should be contextually appropriate. [cite: 600, 601]
    2.  **Refine `_handle_api_error`:**
        * For `google_exceptions.ResourceExhausted`, if `exc.retry_after` is available, prioritize its value for delay. [cite: 602]
        * Explicitly handle `google.generativeai.types.BlockedByPolicyError` (or its current equivalent) and `google.generativeai.types.StopCandidateException` (if generation stops for reasons like safety/recitation) as non-retryable for the current prompt. [cite: 603]
    3.  **Implement Robust `_call_gemini` Response Handling:**
        * If content blocking is detected (via `response.prompt_feedback.block_reason` or `candidate.finish_reason == SAFETY/RECITATION`), raise a specific `GenerationError` (e.g., `ContentBlockedError(GenerationError)`) detailing the block reason and safety ratings. This error should be marked non-retryable. [cite: 604]
        * Fully populate the `LLMResponse` object with `prompt_feedback`, `finish_reason`, and `safety_ratings` from the Gemini API response. [cite: 605]
    4.  **Ensure `_call_gemini` Uses Configured API Parameters:** The `genai.GenerativeModel.generate_content` call must use a `generation_config` object incorporating all relevant parameters from `self.config.GENERATION_CONFIG_DEFAULT`. [cite: 606]
* **Priority:** High.

**Part 6 & 7: Vocabulary Management (VocabularyManager)**
* **Objective:** Ensure robust vocabulary loading, parsing, context-aware retrieval, and accurate usage tracking.
* **Tasks:**
    1.  **Implement Full `_parse_structured_list` (GLCAI Parser):** Verify that all fields specified in User Guide Section 7.2 (word, count -> `usage_count`, layers, categories, source, example, tags) are correctly mapped to `VocabularyItem` attributes. Handle `layers` and `tags` robustly, whether they are comma-separated strings or lists. Ensure `usage_count` is an integer.
    2.  **Refine `_add_item` Layer Assignment:** Confirm correct filtering against `VALID_LAYERS`, use of `_infer_layers` (based on `config.LAYER_KEYWORDS`) as fallback, and assignment of a default layer (e.g., "psychological") if no valid layer is determined. [cite: 611]
    3.  **Refine `get_vocabulary_for_prompt`:** Ensure `avoid_words` comparison is case-insensitive. For style-based weight adjustment, ensure `VocabularyItem.tags` comparisons are case-insensitive or tags are consistently cased. Document expected tag format for styles (e.g., "colloquial_high").
    4.  **Complete `get_vocabulary_by_eti_category`:** Update `eti_category_map_to_vocab_criteria` to map all ETI components from User Guide Section 1.2 to corresponding `VocabularyItem.categories` and/or `tags`. [cite: 614]
* **Priority:** Medium.

**Part 8: Base Evaluator Structure and LLM Evaluator (`Evaluator`)**
* **Objective:** Finalize the core LLM-based evaluation for accurate and structured feedback.
* **Tasks:**
    1.  **Implement Optional JSON Repair in `_parse_evaluation_response`:** Use the `json_repair` library conditionally. Enclose import and usage in `try-except ImportError`. Control activation via `NGGSConfig.ENABLE_JSON_REPAIR_DEFAULT`. [cite: 618, 619]
    2.  **Ensure Completeness in `_validate_and_structure_parsed_data`:** The `expected_score_keys_map` (from `_get_default_scores_and_reasons`) must reflect all score keys in the finalized `DEFAULT_EVALUATION_TEMPLATE` (Part 4). For any missing keys in the LLM's JSON, assign a default score (e.g., 2.0) and a reason like "(LLM評価なし - Missing Key)". [cite: 620, 621]
    3.  **Add Error Context in `Evaluator.evaluate`:** If an LLM call or JSON parsing fails, the returned `EvaluationResult.analysis` field must include a truncated preview of the raw LLM response (e.g., `truncate_text(llm_response_text, 200)`). [cite: 622]
* **Priority:** High.

**Part 9: Evaluator Components (ETI, RI Evaluators)**
* **Objective:** Implement and tune ETI and RI heuristic evaluations according to the User Guide and Roadmap.
* **Tasks:**
    1.  **`ExtendedETIEvaluator.evaluate`:**
        * **ETI Component Mapping:** Ensure LLM score keys used for ETI components (e.g., `base_llm_eval.scores.get("eti_boundary")`) directly match keys in `DEFAULT_EVALUATION_TEMPLATE` (Part 4). If direct LLM evaluation of all ETI components (Boundary, Ambivalence, etc. from User Guide 1.2 [cite: 5]) is not feasible, refine mappings from general LLM scores or implement dedicated simple heuristics for each. [cite: 626, 627]
        * **Refine `_evaluate_phase_transitions_heuristic`:** Incorporate `self.narrative_flow.analyze_phase_distribution()` (Part 10). Penalize significant deviations from `config.PHASE_BALANCE_TARGETS` or undesirable patterns (e.g., excessive consecutive monologues). The score should reflect "naturalness and effectiveness." [cite: 628, 629]
        * **Integrate `_evaluate_subjectivity_heuristic_for_eti`:** This ETI sub-component should leverage the `SubjectiveEvaluator` (Part 10). Replace the simple heuristic with a call to `SubjectiveEvaluator.evaluate()` and use its main score or a relevant sub-score. [cite: 630, 631]
    2.  **`RIEvaluator`:**
        * **Tune `_calc_clarity_score`:** Refine scoring for average sentence length and sentence length variance (CV) per "Evaluation Rubric Improvement" goal (Roadmap). Define ideal CV ranges and penalty functions in `NGGSConfig` if complex. [cite: 632, 633]
        * **Refine `_calc_emotion_flow_score`:** Consider penalties for both underuse and overuse of emotional keywords. [cite: 634]
* **Priority:** High.

**Part 10: Evaluator Components (NarrativeFlowFramework & SubjectiveEvaluator)**
* **Objective:** Implement detailed structural and subjective narration analysis.
* **Tasks:**
    1.  **Enhance `NarrativeFlowFramework.analyze_phase_distribution`:**
        * Implement keyword/pattern-based detection for "live_report" and "serif_prime" phases using new keyword lists from `NGGSConfig` (e.g., `config.LIVE_REPORT_KEYWORDS`, `config.SERIF_PRIME_MARKERS`). [cite: 638]
        * Ensure these are distinct from "narration" and "serif." The returned dictionary must include all phase keys from `config.PHASE_BALANCE_TARGETS`. [cite: 639, 640]
    2.  **Verify `NarrativeFlowFramework.analyze_layer_distribution`:** Ensure it uses `config.LAYER_KEYWORDS` and covers all four layers (physical, sensory, psychological, symbolic) per User Guide 1.2. [cite: 641]
    3.  **Enhance `NarrativeFlowFramework.generate_flow_template`:** Strengthen logic for varied and contextually relevant narrative structure instructions based on theme, `emotion_arc_str`, and `perspective_mode_str`. Allow theme-specific rules to be configurable via `NGGSConfig`. [cite: 642, 643]
    4.  **Complete `SubjectiveEvaluator` Implementation:**
        * `_evaluate_first_person_usage`: Maintain frequency and distribution penalty. [cite: 644]
        * `_evaluate_internal_expression`: Ensure correct use of `config.SUBJECTIVE_INNER_KEYWORDS`. Diversity scoring is key. [cite: 645]
        * `_evaluate_monologue_quality`: Enhance beyond simple patterns. Consider average length and a keyword-based check for "depth." [cite: 646]
        * **Modify `_evaluate_perspective_consistency`:** Accept `perspective_mode_str` as an argument. If it indicates an intentional "perspective_shift," adjust/waive penalties for mixed pronouns. [cite: 647, 648]
* **Priority:** Medium-High.

**Part 11 & 12: TextProcessor (Initialization, Evaluation Helpers, Main Logic Skeleton & Loop 0)**
* **Objective:** Ensure `TextProcessor` correctly orchestrates evaluations and manages Loop 0 robustly.
* **Tasks:**
    1.  **Enhance `_perform_full_evaluation`:**
        * Ensure each `_run_*_eval` call stores the full `EvaluationResult` object (e.g., `full_results_dict["eti_result"] = eti_eval_result_obj`). [cite: 651]
        * The `aggregated_scores_map` must correctly map and store all primary scores from each evaluator. [cite: 652]
    2.  **Implement Derived Score Calculations (`_calculate_*_score` methods):**
        * **`_calculate_emotion_arc_score`:** Implement a concrete heuristic. If `emotion_arc_param` is "A->B->C", check for keywords related to A, B, C in respective text thirds. Score based on presence/ordering. Use emotion keywords from `NGGSConfig`.
        * **`_calculate_colloquial_score`:** Implement a concrete heuristic. Use `config.COLLOQUIAL_MARKERS_INFORMAL` and `config.COLLOQUIAL_MARKERS_FORMAL` (new `NGGSConfig` fields). Calculate informal vs. formal marker ratio and compare against `colloquial_level_param` to derive a score. [cite: 656]
        * Ensure all derived score calculations use parameters from `self.config` accurately. [cite: 657]
    3.  **Refine `TextProcessor.process` Loop 0 Logic:**
        * **NDGS Input Handling:** If `ndgs_input_data_dict` is provided, ensure `self.ndgs_parser.parse()` (Part 15) is called. Its output (`initial_text`, `parameters_override`) must correctly update `current_text_for_processing` and relevant `final_*` style parameters. NDGS-provided parameters should override CLI arguments. [cite: 658, 659]
        * **Initial Generation Prompt:** If `skip_initial_generation_flag` is `False`, `_perform_initial_generation` must use `self.generation_template`, populating it with `current_text_for_processing` (as context), `final_target_text_length`, all style parameters, and `final_narrative_flow_prompt_str_for_gen`. [cite: 660]
        * **State Update:** After Loop 0 evaluation, `best_text_overall`, `best_full_eval_data_overall`, and `results_output_dict["best_text_loop_index"]` must be correctly updated. [cite: 661]
        * **GLCAI Feedback Call (Loop 0):** After text is generated/provided in Loop 0 and evaluated, call `self.glcai_feedback.track_vocabulary_usage(current_text_for_next_loop, self.vocab_manager.items)` if `self.glcai_feedback` is enabled. [cite: 662]
    4.  **Verify `_handle_error` Method:** Ensure `ERROR_SEVERITY_MAP` (Part 2) is strictly applied. [cite: 663]
    5.  **Update `_generate_text` Stub (Interface Alignment):** The stub for `_generate_text` (in Part 12's context, to be fully implemented in Part 14) must match the full signature defined in Part 14 (including `is_improvement`, `evaluation_summary_json_str`, etc.). [cite: 664]
* **Priority:** High.

**Part 13: TextProcessor (Improvement Loop and Instruction Generation)**
* **Objective:** Fully implement the iterative improvement loop and dynamic, targeted instruction generation.
* **Tasks:**
    1.  **Implement Full `_perform_improvement_loop` Functionality:**
        * Implement comparison logic: After new text is generated and evaluated, compare its `overall_quality` (and potentially other key scores from `new_full_eval_data_content.get("aggregated_scores")`) with `current_best_evaluation_data.get("aggregated_scores")`. [cite: 667]
        * If the new score is significantly better (e.g., `new_score > old_best_score + self.config.IMPROVEMENT_SCORE_MARGIN` – add `IMPROVEMENT_SCORE_MARGIN` to `NGGSConfig`), update `best_text_overall`, `best_full_eval_data_overall`, and `results_output_dict["best_text_loop_index"]`. [cite: 668]
        * **GLCAI Feedback Call (Improvement Loop):** After improved text is generated and evaluated within this loop, call `self.glcai_feedback.track_vocabulary_usage(newly_generated_text, self.vocab_manager.items)` if `self.glcai_feedback` is enabled. [cite: 669]
    2.  **Implement `_generate_llm_improvement_instructions` Prompting:**
        * Define a new, dedicated prompt template string in `NGGSConfig` (e.g., `LLM_INSTRUCTION_GENERATION_TEMPLATE`) specifically for asking the LLM to generate improvement instructions. Do not reuse `self.improvement_template`. [cite: 670]
        * This new template should clearly instruct the LLM: "Based on the provided text and its evaluation summary, generate a list of 3-5 concrete, actionable improvement suggestions... Focus on [specific low-scoring areas]." [cite: 671]
        * Populate this template correctly. [cite: 672]
    3.  **Implement Temperature Control:** Ensure the improvement loop temperature adjustment (using `config.IMPROVEMENT_BASE_TEMPERATURE`, `MIN_TEMPERATURE`, `DECREASE_PER_LOOP`) is correctly applied when calling the text generation method from `_perform_improvement_loop`. [cite: 672]
* **Priority:** High.

**Part 14: TextProcessor (Instruction Templates and Text Generation Method - Full Implementation)**
* **Objective:** **This is a critical, highest-priority task.** Complete all specific instruction template generation methods and the main text generation method. [cite: 675]
* **Tasks:**
    1.  **Full Implementation of `_create_*_template` Methods:**
        * For each method (`_create_readability_template`, `_create_subjective_template`, etc.):
            * Implement detailed logic to analyze the provided evaluation results (e.g., `ri_result_obj.scores`). [cite: 675]
            * Identify specific weaknesses related to the strategy (e.g., low `ri_result.scores.get("clarity")` for readability). [cite: 676]
            * Generate a multi-point, actionable instruction string. Instructions must be specific. [cite: 677]
            * Use `self.vocab_manager` to suggest 3-5 relevant vocabulary words for each major instruction point. [cite: 678]
            * Return `Result.ok(instruction_string)`. [cite: 678]
    2.  **Full Implementation of `_generate_text` Method:**
        * Correctly select `self.generation_template` or `self.improvement_template` based on `is_improvement`. [cite: 679]
        * **Initial Generation Mode (`is_improvement == False`):** Populate `self.generation_template` with `input_text_or_context` (as initial context), `target_length`, style parameters, `narrative_flow_section`, and general `vocabulary_list_str`. [cite: 680]
        * **Improvement Generation Mode (`is_improvement == True`):** Populate `self.improvement_template` with all specified parameters: `original_text` (`input_text_or_context`), `evaluation_results_json` (`evaluation_summary_json_str`), `low_score_items_str`, `high_score_items_str`, `layer_distribution_analysis_str`, `phase_distribution_analysis_str`, `improvement_section` (`improvement_section_content_str`), and relevant `vocabulary_list_str`. Ensure all placeholders in `DEFAULT_IMPROVEMENT_TEMPLATE` are correctly populated.
        * Use `SafeDict` for formatting. [cite: 685]
        * Call `self.llm.generate()` with the formatted prompt and correct `temperature_override`. [cite: 685]
    3.  **Implement `TextProcessor._generate_html_report(self, results_data: JsonDict) -> str`:**
        * This new private method takes the full `results_output_dict`. [cite: 686]
        * Generate a comprehensive HTML string including: Job ID, Timestamps, Parameters, a summary section with final scores, a section for each version/loop (loop number, text preview, key scores, distributions, strategy/instructions), and the final best text.
        * In `_finalize_results`, call this method and store the **file path** of the saved HTML report (e.g., `<output_dir>/<job_id>/<job_id>_report.html`) in `results_output_dict["html_report_path"]`. The HTML file itself should be saved by this new method or by `_finalize_results` after getting the content. [cite: 691, 692]
* **Priority:** Highest.

**Part 15: Integration Components (NDGS Parser & GLCAI Feedback)**
* **Objective:** Ensure seamless data exchange with NDGS (input) and GLCAI (output).
* **Tasks:**
    1.  **Activate GLCAI Calls in `TextProcessor.process`:** Uncomment and fully integrate calls to `self.glcai_feedback.track_vocabulary_usage(generated_text, self.vocab_manager.items)` after text generation/evaluation in Loop 0 and within each successful improvement loop iteration. Ensure `self.vocab_manager.items` provides the full list of `VocabularyItem` objects. [cite: 694, 695]
    2.  **Enhance `NDGSIntegration.parse` Parameter Mapping:** Map all relevant NGGS-Lite runtime parameters (e.g., `target_length_override`, `phase_focus_override`, etc. as used in `TextProcessor.process`) from `ndgs_parsed_content.get("parameters_override", {})`. Document this mapping in the method's docstring. [cite: 696]
* **Priority:** High.

**Part 16: Batch Processing Class (BatchProcessor)**
* **Objective:** Ensure robust and informative batch processing, including NDGS file handling and HTML report linking.
* **Tasks:**
    1.  **Implement NDGS Handling in `BatchProcessor._process_single_file`:** When `file_path_obj.suffix.lower() == ".json"`, correctly call `self.processor.ndgs_parser.parse_from_file(file_path_obj)`. The resulting `ndgs_data_for_processor` (a `JsonDict`) must be passed to `self.processor.process(..., ndgs_input_data_dict=ndgs_data_for_processor)`. [cite: 698]
    2.  **Implement HTML Report Links in `_generate_html_batch_summary_report`:** Create correct relative links (e.g., `./{safe_job_id}/{safe_job_id}_report.html`) to individual job HTML reports, using the `job_results.get("html_report_path")` (set in Part 14). [cite: 699]
* **Priority:** Medium.

**Part 17: Main Execution Function and Entry Point (main, parse_arguments)**
* **Objective:** Create a fully functional CLI and robust main execution flow.
* **Tasks:**
    1.  **Ensure Accurate Default Values in `parse_arguments` Help:** For `default=None` arguments that derive defaults from `NGGSConfig` (e.g., `--perspective`), use `getattr(temp_config_for_help, 'CONFIG_ATTRIBUTE_NAME', fallback_value)` to display correct defaults. Match attribute names with those added in Part 1. [cite: 702, 703]
    2.  **Modify `setup_components` Template Loading:** Use `load_template(..., config=effective_config)` for all templates. Remove `_DUMMY_DEFAULT_TEMPLATES`. `load_template` should use `effective_config` for file paths or fall back to global `DEFAULT_TEMPLATES` (per Part 4 revision). [cite: 703]
    3.  **Handle HTML Report Path in `run_single_job`:** Expect `job_result_dict["html_report_path"]` (from Part 14) to be a file path string. Log this path. Saving the HTML file is `TextProcessor`'s responsibility. [cite: 704, 705]
* **Priority:** High.

**Part 18: Script Entry Point**
* **Objective:** Finalize script startup, shutdown, and conditional module availability checks.
* **Tasks:**
    1.  **Confirm `CONCURRENT_FUTURES_AVAILABLE` Definition:** Ensure this flag is correctly defined in Part 1. [cite: 708]
    2.  **Refine Logger Initialization:** Remove `entry_point_logger`. Rely solely on the `logger` instance configured at the start of `main()` (Part 17) and fully by `setup_logging()`. Ensure signal handlers use this global `logger`. [cite: 709, 710]
    3.  **Ensure `JSON_REPAIR_AVAILABLE` Flag Usage:** This flag (set in Part 18) must be correctly referenced in `Evaluator._parse_evaluation_response` (Part 8) to conditionally enable `json_repair`. [cite: 711]
    4.  **Final Import Review:** Perform a final review to ensure all imports are in Part 1 and no circular dependencies exist.
* **Priority:** Medium.

## 4. 📌 Constraints and Non-Goals
* **Python 3.x Adherence:** Strictly Python 3.x. Utilize modern Python 3 features. [cite: 713, 714]
* **No New External Dependencies:** Beyond `google-generativeai` and optionally `json_repair`, do not add new third-party libraries without approval. (`concurrent.futures` is standard). [cite: 715, 716]
* **User Guide as Primary Specification:** Functional implementation must align with the User Guide. If discrepancies arise with this SoW, prioritize the User Guide and flag the issue. [cite: 717]
* **Backend Logic Focus:** No GUI development. This task is for script finalization. [cite: 718]
* **Performance Optimization Secondary:** Correctness and completeness are prioritized over extensive performance profiling for v1.8, unless a specific bottleneck is identified. [cite: 719]

## 5. 📋 Expected Output Format
* **Primary Deliverable:** A single, fully executable Python script: `NGGS-Lite_v1.8_Jules_Implemented.py`. [cite: 721]
* **Internal Structure:**
    * Shebang: `#!/usr/bin/env python3`
    * Encoding: `# -*- coding: utf-8 -*-`
    * Script-level docstring.
    * Clear Part demarcations (`# === Part X: Description ===`). [cite: 723]
    * All imports in Part 1. [cite: 723]
    * Comprehensive Google-style docstrings. [cite: 723]
    * Full type hinting. [cite: 724]
    * Concise comments (English or Japanese) for complex logic. [cite: 724]
* **No External Core Configuration Files:** Core config via `NGGSConfig` defaults and CLI overrides. Template and vocabulary files are external. [cite: 725]

## 6. ✅ Validation and Acceptance Criteria
* **Static Analysis:** Passes `flake8` and `mypy` (strict if possible) without significant errors/warnings. [cite: 727]
* **Compilation:** `python -m py_compile NGGS-Lite_v1.8_Jules_Implemented.py` succeeds. [cite: 728]
* **Execution & Functional Verification:**
    * Runs without Python exceptions with valid sample inputs (text file, direct text, NDGS JSON, batch). [cite: 729]
    * All CLI arguments (Part 17) are functional. [cite: 730]
    * Core text generation produces Gothic-style text.
    * All evaluation metrics (ETI, RI, Subjective, Derived) are calculated and present in output JSON. [cite: 731]
    * Iterative improvement loop functions and selects the best version. [cite: 732]
    * NDGS input correctly influences execution. [cite: 733]
    * GLCAI feedback JSON is correctly generated. [cite: 734]
    * HTML reports (individual and batch summary) are generated with correct paths. [cite: 735, 736]
* **Output Verification:** JSON outputs are well-formed and complete. Final text is saved. [cite: 736, 737]

## 7. 🔄 Feedback and Iteration Protocol
* Feedback will reference specific Part numbers and method/class names from this SoW. [cite: 738]
* Address all feedback points for re-submission. [cite: 741]
* Iteration continues until all requirements are met. [cite: 742]

## 8. 📅 Schedule and Priority (Reference SoW v3.0)
* **Overall Deadline:** 3 weeks from SoW acceptance.
* **Phase 1 (Week 1 - Critical Path & Core Implementation):** Part 14 (Templates, GenText), Part 1 (NGGSConfig), Part 11/12 (GLCAI calls, Derived Scores), Part 15 (NDGS map, GLCAI calls).
* **Phase 2 (Week 2 - Evaluation Refinement & Loop Finalization):** Part 9 (ETI/RI heuristics), Part 13 (Improvement Loop, LLM Instructions), Part 5/8 (LLMClient errors, Evaluator json_repair).
* **Phase 3 (Week 3 - Integrations, Batch, Final Polish & Testing):** Part 10 (Phase/Layer detect, Subjective eval), Part 6/7 (Vocab style/ETI map), Part 16 (Batch NDGS/HTML), Part 17/18 (CLI, main, entry point), Comprehensive Testing.

## 9. 🙏 Final Request to You, Jules
Your primary directive is to produce a **fully executable, robust, and feature-complete** `NGGS-Lite_v1.8_Jules_Implemented.py` script that strictly adheres to this Statement of Work and all referenced documents (User Guide, Roadmap, "Final Check Document" items as incorporated herein).

**Key Focus Areas:**
1.  **Implement All Stubbed/Unimplemented Functionalities:** Especially the `_create_*_template` methods (Part 14) and the core improvement loop logic within `TextProcessor` (Part 13). [cite: 755]
2.  **Ensure User Guide Feature Completeness:** All User Guide features must be operational, including all evaluation metrics and external system integrations (NDGS, GLCAI). [cite: 756]
3.  **Resolve "Final Check Document" Items:** Address all issues identified (as translated into tasks within this SoW). [cite: 757]
4.  **Achieve Zero Runtime Errors:** Under normal operating conditions with valid inputs. [cite: 757]
5.  **Maintain High Code Quality:** PEP8 compliance, comprehensive type hinting, and clear documentation. [cite: 758]

If ambiguities arise or technical constraints prevent exact adherence to a minor specification, implement the most robust and logical solution, clearly documenting your reasoning and any deviations in your submission notes or code comments. Your expertise in creating high-quality, production-ready code is crucial for this project's success. [cite: 759]
