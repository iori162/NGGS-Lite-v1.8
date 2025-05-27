NGGS-Lite v1.8.py: Full Refactoring, Implementation, and Optimization - Statement of Work (for AI Agent jules)
Document Version: 3.0 (For jules, English Edition)
Project: NGGS-Lite v1.8 Full Implementation
Target Script: NGGS-Lite v1.8 (Refactoring Edition).py (The complete 18-part Python script previously generated and reviewed)
Date: May 27, 2025

1. 🎯 Overall Project Goal
1.1. Primary Objective:
To transform the provided NGGS-Lite v1.8 (Refactoring Edition).py script into a fully executable, robust, and feature-complete Python application. This involves rectifying all identified syntactic and logical issues, implementing all stubbed or missing functionalities as per the "NGGS-Lite v1.8 (Gemini Edition) User Guide," and optimizing the codebase for performance, maintainability, and extensibility.

1.2. Key Deliverables & Success Criteria:

Full Executability: The final script must run without any syntax, indentation, name resolution, or type errors from start to finish when executed with Python 3.x.
Complete Feature Implementation: All functionalities specified in the User Guide must be fully implemented and operational. This explicitly includes, but is not limited to:
ETI (Existential Tremor Index) and RI (Readability Index) evaluations, including all their components.
The Phase Transition Model and Four-Layer Structure analysis.
Emotion Arc score calculation and Colloquial/Gothic Blend score calculation.
NDGS (Neo Dialogue Generation System) input parsing.
GLCAI (Gothic Lexicon Creation AI) vocabulary feedback data generation and output.
Fully implemented TextProcessor._create_*_template methods (for all defined improvement strategies).
A fully functional TextProcessor.process method, including the iterative improvement loop.
Generation of HTML reports as specified, with paths stored in the results dictionary.
Adherence to Design Principles:
Emulate the stable modular structure, robust error handling (custom exceptions, Result type), and data validation practices of dialogue_generator.py (NDGS v4.9α).
Strictly follow the "Stand-alone Priority" and "Quality Concentration Strategy" outlined in the "System Development Roadmap (v2.0)."
Resolution of Identified Issues: All unimplemented sections, stubbed methods, and specific issues pinpointed in the "NGGS-Lite v1.8 (Refactoring Edition) Final Check and Commentary" document (provided as context for this SoW) must be addressed and resolved.
Code Quality:
Strict PEP8 compliance.
Comprehensive type hinting for all functions, methods, and significant variables.
Clear, concise, and informative docstrings (Google style preferred) for all classes, methods, and functions.
Well-placed comments in Japanese or English to clarify complex logic.
1.3. Rationale for this Request:
The current target script, while structurally improved, contains significant unimplemented or stubbed functionalities, particularly in the TextProcessor's improvement loop, instruction template generation, and external system integrations (NDGS/GLCAI). Furthermore, configurations and specific evaluation heuristics require finalization. This SoW provides the detailed specifications necessary to bring the script to a fully operational and high-quality state.

2. 🧾 Context and Provided Materials
2.1. Essential Reference Documents:

NGGS-Lite v1.8 (Gemini Edition) User Guide.rtf (referred to as User Guide)
System Development Roadmap (v2.0).txt (referred to as Roadmap)
dialogue_generator.py (NDGS v4.9α) (referred to as NDGS Script) - For structural and design pattern reference.
NGGS-Lite v1.8 (Refactoring Edition).py (The 18-part consolidated script that is the Target Script for this SoW).
NGGS-Lite v1.8 (Refactoring Edition) Final Check and Commentary (referred to as Final Check Document) - This document contains critical feedback and a to-do list that must be actioned.
2.2. Current State of the Target Script (based on Final Check Document):

Implemented: Core class skeletons (NGGSConfig, TextProcessor, LLMClient, Evaluators, VocabularyManager, NDGSIntegration, GLCAIVocabularyFeedback, BatchProcessor), basic error handling (NGGSError, Result), logging setup, fundamental utilities, template loading, and vocabulary parsing.
Key Unresolved Issues & Unimplemented Functionality (To be addressed by Jules):
Critical Unimplemented Methods: All TextProcessor._create_*_template methods (Part 14) are currently stubs.
Incomplete Core Logic: The TextProcessor.process method's improvement loop (Part 13) is not fully functional (comparison of new vs. best evaluations, and subsequent logic). The TextProcessor._generate_text method (Part 14) needs full implementation for both initial and improvement generation modes.
Missing Integrations: GLCAI feedback (track_vocabulary_usage) calls are not activated within TextProcessor.process. NDGS parameter mapping from parsed data to TextProcessor runtime parameters is not fully detailed.
Configuration Gaps: NGGSConfig (Part 1) is missing GEMINI_SAFETY_SETTINGS, log rotation parameters, and some batch/style default parameters.
Evaluation Logic: Derived scores for Emotion Arc and Colloquial/Gothic Blend (_calculate_emotion_arc_score, _calculate_colloquial_score in Part 11) are placeholders. ETI/RI heuristic details (Part 9) and Phase Detection in NarrativeFlowFramework (Part 10) require refinement based on User Guide/Roadmap.
Reporting: HTML report generation (Part 14, via TextProcessor._finalize_results) is a stub.
Code Hygiene: Import statements need consolidation in Part 1.
2.3. Execution Environment:

Python 3.x (ensure compatibility with features used, e.g., dataclasses, pathlib, typing).
Google Gemini API (via google-generativeai library).
Optional dependencies (json_repair, concurrent.futures) should be handled conditionally.
3. 📝 Detailed Requirements (Per Part)
3.1. General Directives for Jules:

Prioritize Final Check Document: Meticulously review and action every point, unimplemented item, and suggested improvement detailed in the "Final Check Document." This is paramount.
Emulate NDGS Design: Where applicable, adopt the robust modular design, exception handling patterns, data validation techniques, and clear process flows demonstrated in the NDGS Script.
Zero-Error Mandate: The final script must be free of all syntax, indentation, runtime, and type errors. Verification via python -m py_compile your_script_name.py and mypy your_script_name.py should pass without issues. pytest should be usable for unit testing key components.
User Guide Adherence: All functionalities must align precisely with the descriptions and specifications in the User Guide (e.g., ETI components, Phase Transition Model, Four-Layer Structure definitions, NDGS/GLCAI I/O formats).
Roadmap Alignment: Implementations should reflect the "Quality Concentration Strategy," enhancing aspects like phase transition naturalness, four-layer expressiveness, and subjective viewpoint immersion.
3.2. Specific Part-by-Part Implementation & Refinement Instructions:

Part 1: Imports, Constants, Core Configuration (NGGSConfig)
Objective: Finalize the foundational configuration class and global constants, ensuring all system parameters are centrally managed and correctly defaulted.
Specific Tasks:
Add to NGGSConfig:
GEMINI_SAFETY_SETTINGS: Dict[str, str] = field(default_factory=lambda: {"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}) (or other appropriate defaults based on Gemini API documentation).
LOG_MAX_BYTES_DEFAULT_VAL: int = 5 * 1024 * 1024 (5MB).
LOG_BACKUP_COUNT_DEFAULT_VAL: int = 3.
BATCH_MAX_WORKERS_DEFAULT: int = 1 (or calculate based on os.cpu_count()).
BATCH_FILE_PATTERN_DEFAULT: str = "*.txt".
PERSPECTIVE_MODE_DEFAULT: str = "subjective_first_person".
PHASE_FOCUS_DEFAULT: str = "balanced".
COLLOQUIAL_LEVEL_DEFAULT: str = "medium".
NARRATIVE_THEME_DEFAULT: str = "記憶回帰型" (Memory Regression Type - an example, can be refined).
ENABLE_NDGS_PARSER_DEFAULT: bool = True.
ENABLE_GLCAI_FEEDBACK_DEFAULT: bool = True.
ENABLE_JSON_REPAIR_DEFAULT: bool = False.
Consolidate Imports: Move all import statements to the beginning of this Part 1. Organize them into standard library, third-party, and then project-specific (none in this case for a single file). Remove duplicates and unused imports. Ensure typing imports are comprehensive (e.g., TypeAlias).
Define CONCURRENT_FUTURES_AVAILABLE: Define this boolean flag based on a try-except ImportError for concurrent.futures immediately after standard library imports.
Remove the dummy ConfigurationError definition from Part 1; it will be formally defined in Part 2.
Priority: High (Critical for system stability and configuration).
Dependencies: Affects Part 3 (Logging), Part 4 (LLMClient Init), Part 16 (BatchProcessor), Part 17 (Argparse Defaults).
Part 2: Core Exceptions and Result Type
Objective: Solidify the custom error handling framework.
Specific Tasks:
NGGSError.details Enrichment: In all locations where NGGSError or its subclasses are instantiated, ensure the details dictionary is populated with contextually relevant information (e.g., failed_component, operation, input_preview, original_exception_type, original_exception_message). Refer to NGGSError docstring for suggested keys.
LLMResponse.metadata Population: Modify LLMClient._call_gemini (Part 5) to populate the metadata field of the LLMResponse object with API call duration, token usage (prompt, candidates, total), and any other relevant API-returned metadata.
Priority: Medium (Existing implementation is robust, but enrichment improves debuggability).
Dependencies: Affects all parts that raise or handle custom exceptions, especially Part 5 (LLMClient).
Part 3: Utility Functions (Logging, Text, File I/O)
Objective: Ensure all utility functions are robust and fully configurable.
Specific Tasks:
setup_logging Enhancement: Modify to use LOG_MAX_BYTES_DEFAULT_VAL and LOG_BACKUP_COUNT_DEFAULT_VAL from the NGGSConfig instance for RotatingFileHandler parameters.
get_metric_display_name Expansion: Review the display_map and ensure it comprehensively covers all keys present in the updated DEFAULT_EVALUATION_TEMPLATE (from Part 4) and all relevant NGGSConfig parameter keys that might be displayed to the user or in reports.
CompactJSONEncoder for Enum: Add logic to default() method to serialize Enum objects using their value attribute (i.e., if isinstance(obj, Enum): return obj.value).
Priority: Medium (Logging and metric display are important for usability).
Dependencies: Part 1 (NGGSConfig for logging parameters).
Part 4: Utility Functions (Continued), LLM Client (Initialization)
Objective: Finalize template management and LLM client initialization, particularly Gemini client setup.
Specific Tasks:
LLMClient._initialize_gemini with Safety Settings: Ensure this method correctly retrieves GEMINI_SAFETY_SETTINGS from self.config (added in Part 1) and passes them to genai.GenerativeModel(..., safety_settings=...).
Default Template Content Finalization: Critically review and finalize the content of DEFAULT_GENERATION_TEMPLATE, DEFAULT_EVALUATION_TEMPLATE, and DEFAULT_IMPROVEMENT_TEMPLATE. Ensure DEFAULT_EVALUATION_TEMPLATE's JSON output structure explicitly lists all evaluation criteria detailed in User Guide Section 1.2 (including all ETI components, RI components, etc., that are expected to be scored by the LLM).
load_template Call Signature: Confirm that load_template is called from setup_components (Part 17) with the config instance, not the global DEFAULT_TEMPLATES dictionary directly, to ensure file-based templates are prioritized correctly using config.DEFAULT_TEMPLATES_DIR.
Priority: High (Correct LLM initialization and effective prompts are core to functionality).
Dependencies: Part 1 (NGGSConfig), Part 5 (LLMClient generation logic), Part 8 (Evaluator using templates), Part 17 (setup_components).
Part 5: LLM Client (Generation Logic)
Objective: Create a highly resilient LLM interaction layer.
Specific Tasks:
_setup_mock Full Coverage: Update the JSON structure returned by the mock evaluation to include all keys defined in the finalized DEFAULT_EVALUATION_TEMPLATE (from Part 4). Ensure mock scores are within a realistic range (e.g., 1.0-5.0) and reasons are contextually appropriate (e.g., "Mock reasoning for 'eti_boundary'").
_handle_api_error Precision:
For google_exceptions.ResourceExhausted, if exc has a retry_after attribute, prioritize its value for the delay.
Explicitly handle google_exceptions.BlockedByPolicy (if applicable, check current Gemini API for this or similar) and potential google.generativeai.types.StopCandidateException (or similar indicating generation stopped by non-error reasons like safety/recitation) as non-retryable for the current prompt.
_call_gemini Robust Response Handling:
When a content block is detected (via response.prompt_feedback.block_reason or candidate.finish_reason == SAFETY/RECITATION), raise a specific GenerationError (e.g., ContentBlockedError subclass of GenerationError) detailing the block reason and safety ratings. This error should be marked as non-retryable by _handle_api_error.
Populate the LLMResponse object with prompt_feedback, finish_reason, and safety_ratings from the Gemini API response.
_call_gemini API Parameters: Ensure genai.GenerativeModel.generate_content is called with a generation_config object that incorporates all relevant parameters from self.config.GENERATION_CONFIG_DEFAULT (temperature, top_p, top_k, max_output_tokens).
Priority: High (Stable LLM communication is critical).
Dependencies: Part 2 (LLMResponse, GenerationError), Part 4 (LLMClient init).
Part 6 & 7: Vocabulary Management (VocabularyManager)
Objective: Ensure robust vocabulary loading, processing, and context-aware retrieval.
Specific Tasks:
_parse_structured_list (GLCAI Parser): Verify that all fields specified in User Guide Section 7.2 for GLCAI output (word, count -> usage_count, layers, categories, source, example, tags) are correctly mapped to VocabularyItem attributes. Implement robust handling for layers and tags if they come as comma-separated strings or lists of strings. Ensure usage_count is an integer.
_add_item Layer Assignment: Confirm that layer assignment correctly filters against VALID_LAYERS, uses _infer_layers (based on config.LAYER_KEYWORDS) as a fallback, and assigns a default layer (e.g., "psychological") if no valid layer is determined.
get_vocabulary_for_prompt Refinements:
Ensure avoid_words list comparison is case-insensitive.
For style-based weight adjustment, ensure VocabularyItem.tags are consistently cased or comparisons are case-insensitive. Document expected tag format for styles (e.g., "colloquial_high", "formal_low", etc.).
get_vocabulary_by_eti_category Completeness: Update eti_category_map_to_vocab_criteria to map all ETI components from User Guide Section 1.2 (Boundary, Ambivalence, Transcendental Violation, Uncertainty, Internal Transformation, Phase Transition, Subjectivity) to corresponding VocabularyItem.categories and/or tags.
Priority: Medium (Core vocabulary functionality is present; this is refinement).
Dependencies: Part 1 (NGGSConfig for keywords/paths), User Guide (for GLCAI format, ETI components).
Part 8: Base Evaluator Structure and LLM Evaluator
Objective: Finalize the LLM-based evaluation core for accurate and structured feedback.
Specific Tasks:
_parse_evaluation_response JSON Repair: Implement the optional JSON repair logic using the json_repair library. Enclose the import and usage in a try-except ImportError block, and control its activation via NGGSConfig.ENABLE_JSON_REPAIR_DEFAULT (added in Part 1).
_validate_and_structure_parsed_data Completeness: Ensure the expected_score_keys_map (derived from _get_default_scores_and_reasons) reflects all score keys defined in the finalized DEFAULT_EVALUATION_TEMPLATE (from Part 4). For any missing keys in the LLM's JSON response, assign a clear default score (e.g., 2.0) and a reason like "(LLM評価なし - Missing Key)" to validated_reasons.
Evaluator.evaluate Error Context: If LLM call or JSON parsing fails, ensure the returned EvaluationResult.analysis field includes a truncated preview of the raw LLM response (truncate_text(llm_response_text, 200)).
Priority: High (Reliable LLM evaluation is key to the improvement loop).
Dependencies: Part 3 (truncate_text), Part 4 (Templates, validate_template), Part 5 (LLMClient).
Part 9: Evaluator Components (ETI, RI Evaluators)
Objective: Implement and tune ETI and RI heuristic evaluations.
Specific Tasks:
ExtendedETIEvaluator.evaluate:
ETI Component Mapping: Ensure LLM score keys used to derive ETI components (e.g., base_llm_eval.scores.get("eti_boundary")) directly match the keys defined in DEFAULT_EVALUATION_TEMPLATE (Part 4). If direct LLM evaluation of ETI components is not feasible, refine the mapping from more general LLM scores (e.g., gothic_atmosphere) or implement dedicated simple heuristics for each ETI component (Boundary, Ambivalence, etc., as per User Guide 1.2).
_evaluate_phase_transitions_heuristic Refinement: Enhance this heuristic by incorporating self.narrative_flow.analyze_phase_distribution() (from Part 10). For example, penalize if the distribution significantly deviates from config.PHASE_BALANCE_TARGETS or if undesirable patterns (e.g., excessive consecutive monologues without narration/dialogue) are detected. The score should reflect "naturalness and effectiveness."
_evaluate_subjectivity_heuristic_for_eti Integration: This ETI sub-component should ideally leverage the more detailed SubjectiveEvaluator (Part 10). Consider replacing this simple heuristic with a call to SubjectiveEvaluator.evaluate() and extracting its main score, or a relevant sub-score, for ETI calculation.
RIEvaluator:
_calc_clarity_score Tuning: Refine the scoring curve for average sentence length and sentence length variance (CV) based on the "Evaluation Rubric Improvement" goal (Roadmap). Define ideal CV ranges and penalty functions in NGGSConfig if they become complex.
_calc_emotion_flow_score Refinement: Consider both underuse and overuse of emotional keywords for penalty.
_calc_cognitive_load_score Future Consideration: While full NLP for passive voice/complex syntax is beyond v1.8, add comments indicating these as future enhancement points for this heuristic.
Priority: High (Core evaluation metrics).
Dependencies: Part 8 (BaseEvaluator, EvaluationResult), Part 10 (NarrativeFlowFramework, SubjectiveEvaluator), User Guide (ETI/RI definitions), Roadmap (evaluation refinement goals).
Part 10: Evaluator Components (NarrativeFlowFramework & SubjectiveEvaluator)
Objective: Implement detailed structural and subjective narration analysis.
Specific Tasks:
NarrativeFlowFramework.analyze_phase_distribution Enhancement:
Implement keyword-based or simple pattern-based detection for "live_report" and "serif_prime" phases, using new keyword lists from NGGSConfig (e.g., config.LIVE_REPORT_KEYWORDS, config.SERIF_PRIME_MARKERS). Ensure these are distinct from "narration" and "serif" respectively, or document clear rules for inclusion/exclusion if overlap occurs.
The returned dictionary must include all phase keys defined in config.PHASE_BALANCE_TARGETS.
NarrativeFlowFramework.analyze_layer_distribution: Ensure it uses config.LAYER_KEYWORDS and covers all four layers (physical, sensory, psychological, symbolic) as per User Guide 1.2.
NarrativeFlowFramework.generate_flow_template Enhancement: Strengthen the logic to generate more varied and contextually relevant narrative structure instructions based on theme, emotion_arc_str, and perspective_mode_str. Consider allowing theme-specific instruction rules to be configurable via NGGSConfig.
SubjectiveEvaluator Full Implementation:
_evaluate_first_person_usage: Maintain frequency and distribution penalty. Add comments for future enhancements (e.g., analyzing pronoun placement, relation to emotional expressions).
_evaluate_internal_expression: Ensure it correctly uses config.SUBJECTIVE_INNER_KEYWORDS. Diversity scoring is crucial.
_evaluate_monologue_quality: Enhance beyond () and introspection patterns. Consider average monologue length, and a simple keyword-based check for "depth" of content within monologues.
_evaluate_perspective_consistency: Modify to accept perspective_mode_str (e.g., from TextProcessor parameters) as an argument. If perspective_mode_str indicates an intentional "perspective_shift," adjust or waive penalties for mixed pronoun usage.
Priority: Medium-High (Crucial for NGGS-Lite's unique features).
Dependencies: Part 1 (NGGSConfig for keywords), Part 6 (VocabularyManager if used for keyword sources), User Guide (feature definitions).
Part 11 & 12: TextProcessor (Initialization, Evaluation Helpers, Main Logic Skeleton & Loop 0)
Objective: Ensure TextProcessor correctly orchestrates evaluation and manages the Loop 0 process.
Specific Tasks:
_perform_full_evaluation Enhancement:
Ensure each call to _run_*_eval stores the full EvaluationResult object in full_results_dict (e.g., full_results_dict["eti_result"] = eti_eval_result_obj).
The aggregated_scores_map must correctly map and store all primary scores from each evaluator (e.g., agg_scores_map["eti_total_calculated"] = eti_eval_result_obj.scores.get("eti_total_calculated")).
Derived Score Calculation (_calculate_*_score methods) Implementation:
_calculate_emotion_arc_score: Implement a more concrete heuristic. Example: If emotion_arc_param is "A->B->C", check for keywords related to A in the first third, B in the middle, C in the last third. Score based on presence and ordering. Use emotion-related keywords from NGGSConfig.
_calculate_colloquial_score: Implement a more concrete heuristic. Use config.COLLOQUIAL_MARKERS_INFORMAL and config.COLLOQUIAL_MARKERS_FORMAL (new NGGSConfig fields). Calculate ratio of informal vs. formal markers and compare against colloquial_level_param ("high", "medium", "low") to derive a score.
Ensure all derived score calculations use parameters from self.config (target values, penalty factors) accurately.
TextProcessor.process Loop 0 Logic:
NDGS Input: If ndgs_input_data_dict is provided, ensure self.ndgs_parser.parse() (Part 15) is called and its output (initial_text, parameters_override) correctly updates current_text_for_processing and relevant final_* style parameters. NDGS-provided parameters should take precedence over CLI arguments if both exist for the same setting.
Initial Generation Prompt: If skip_initial_generation_flag is False, _perform_initial_generation must use self.generation_template, populating it with current_text_for_processing (as context), final_target_text_length, all style parameters, and the final_narrative_flow_prompt_str_for_gen.
State Update: After Loop 0 evaluation, best_text_overall, best_full_eval_data_overall, and results_output_dict["best_text_loop_index"] must be correctly updated.
GLCAI Feedback Call: Crucially, after text is generated/provided in Loop 0 and evaluated, ensure self.glcai_feedback.track_vocabulary_usage(current_text_for_next_loop, self.vocab_manager.items) is called if self.glcai_feedback is enabled.
_handle_error Method: Ensure ERROR_SEVERITY_MAP (from Part 2) is strictly applied.
_generate_text Stub (in Part 12 context): The stub in Part 12 needs to be updated to reflect the full signature defined in Part 14, including is_improvement, evaluation_summary_json_str, etc. For Part 12's purpose, it can still have minimal internal logic, but the interface must match Part 14.
Priority: High (Core processing workflow).
Dependencies: Part 10 (NarrativeFlowFramework for prompt), Part 14 (_generate_text full signature), Part 15 (NDGSIntegration, GLCAIVocabularyFeedback).
Part 13: TextProcessor (Improvement Loop and Instruction Generation)
Objective: Fully implement the iterative improvement loop and dynamic instruction generation.
Specific Tasks:
_perform_improvement_loop Full Functionality:
Implement the comparison logic: After new text is generated and evaluated in a loop, compare its overall_quality (and potentially other key scores from new_full_eval_data_content.get("aggregated_scores")) with current_best_evaluation_data.get("aggregated_scores").
If the new score is significantly better (e.g., new_score > old_best_score + self.config.IMPROVEMENT_SCORE_MARGIN - add IMPROVEMENT_SCORE_MARGIN to NGGSConfig), update best_text_overall, best_full_eval_data_overall, and results_output_dict["best_text_loop_index"].
GLCAI Feedback Call: After improved text is generated and evaluated within this loop, ensure self.glcai_feedback.track_vocabulary_usage(newly_generated_text, self.vocab_manager.items) is called if self.glcai_feedback is enabled.
_generate_llm_improvement_instructions Prompting:
Create a new, dedicated prompt template string (e.g., LLM_INSTRUCTION_GENERATION_TEMPLATE in NGGSConfig) specifically for asking the LLM to generate improvement instructions, rather than reusing self.improvement_template which is for generating improved text.
This new template should clearly state: "Based on the provided text and its evaluation summary, generate a list of 3-5 concrete, actionable improvement suggestions for the author to enhance the text's quality according to the identified weaknesses. Focus on [specific low-scoring areas if any]."
Populate this template with current_text_content, evaluation_summary_json_str, low_score_items_summary_str, etc.
Temperature Control: Ensure the improvement loop temperature adjustment (using config.IMPROVEMENT_BASE_TEMPERATURE, MIN_TEMPERATURE, DECREASE_PER_LOOP) is correctly applied when calling _generate_text (or llm.generate if _generate_text is not yet fully implemented for improvement in this part's context) from _perform_improvement_loop.
Priority: High (Core iterative refinement mechanism).
Dependencies: Part 12 (Loop 0 setup), Part 14 (_create_*_template full implementation, _generate_text full implementation).
Part 14: TextProcessor (Instruction Templates and Text Generation Method - Full Implementation)
Objective: Complete all specific instruction template generation methods and the main text generation method. This is a critical, highest-priority task.
Specific Tasks:
_create_*_template Methods Full Implementation:
For each stubbed method (_create_readability_template, _create_subjective_template, _create_phase_template, _create_layer_template, _create_emotion_template, _create_colloquial_template, _create_balance_template):
Implement detailed logic to analyze the provided evaluation results (e.g., ri_result_obj.scores, subjective_result_obj.components).
Identify specific weaknesses related to the strategy (e.g., if ri_result.scores.get("clarity") is low for readability strategy).
Generate a multi-point, actionable instruction string. Instructions should be specific (e.g., "Clarify sentence structure by shortening sentences around paragraph X," "Increase sensory details related to 'smell' in the first scene.").
Use self.vocab_manager.get_vocabulary_for_prompt() or get_vocabulary_by_eti_category() to suggest 3-5 relevant vocabulary words for each major point of instruction.
Return Result.ok(instruction_string).
_generate_text Method Full Implementation:
Correctly select self.generation_template or self.improvement_template based on the is_improvement flag.
Initial Generation Mode (is_improvement == False): Populate self.generation_template with input_text_or_context (as initial context), target_length, style parameters (perspective_mode, etc.), narrative_flow_section, and a general vocabulary_list_str.
Improvement Generation Mode (is_improvement == True): Populate self.improvement_template with:
original_text: input_text_or_context (the previous best text).
evaluation_results_json: evaluation_summary_json_str.
low_score_items_str: low_score_items_summary_str.
high_score_items_str: high_score_items_summary_str.
layer_distribution_analysis: layer_distribution_analysis_str.
phase_distribution_analysis: phase_distribution_analysis_str.
improvement_section: improvement_section_content_str (the instructions generated by _create_*_template or LLM).
vocabulary_list_str: Vocabulary relevant to the improvement strategy (this might need to be generated within _perform_improvement_loop and passed, or _generate_text could infer from improvement_section_content_str). Clarification: _perform_improvement_loop should prepare strategy-specific vocabulary and pass it if needed, or improvement_section_content_str should already include vocab suggestions.
Use SafeDict to format the chosen template.
Call self.llm.generate() with the fully formatted prompt and the temperature_override (which would be the adjusted temperature for improvement loops, or default for initial).
TextProcessor._finalize_results HTML Report Generation:
Implement a new private method, e.g., _generate_html_report(self, results_data: JsonDict) -> str, that takes the full results_output_dict.
This method should generate a comprehensive HTML string. The report should include:
Job ID, Timestamps, Parameters used.
A summary section with overall scores (e.g., from final_aggregated_scores).
A section for each "version" (loop iteration) from results_data["versions_data"], showing:
Loop number.
Generated text preview (truncate_text).
Key evaluation scores for that version (overall, ETI, RI, subjective, derived scores).
Phase and Layer distribution summaries.
Improvement strategy and instructions applied (if applicable).
The final best generated text.
In _finalize_results, call this method and store the file path of the saved HTML report in results_output_dict["html_report_path"]. The HTML file should be saved in the job-specific output directory (e.g., <output_dir>/<job_id>/<job_id>_report.html).
Priority: Highest (Core functionality completion).
Dependencies: All previous TextProcessor methods, VocabularyManager, LLMClient.
Part 15: Integration Components (NDGS Parser & GLCAI Feedback)
Objective: Ensure seamless data exchange with NDGS (input) and GLCAI (output).
Specific Tasks:
TextProcessor.process GLCAI Call Activation: Uncomment and fully integrate the calls to self.glcai_feedback.track_vocabulary_usage(generated_text, self.vocab_manager.items) after text generation/evaluation in Loop 0 and within each successful improvement loop iteration. Ensure self.vocab_manager.items provides the full list of VocabularyItem objects.
NDGSIntegration.parse Parameter Mapping: Enhance to map more keys from ndgs_parsed_content.get("parameters_override", {}) to the corresponding NGGS-Lite runtime parameters used in TextProcessor.process (e.g., target_length_override, phase_focus_override, colloquial_level_override, emotion_arc_override, max_loops_override, improvement_threshold_override). Document this mapping clearly in the method's docstring.
Priority: High (Key integration points).
Dependencies: Part 6&7 (VocabularyManager.items), Part 12 (TextProcessor.process structure).
Part 16: Batch Processing Class (BatchProcessor)
Objective: Ensure robust and informative batch processing.
Specific Tasks:
BatchProcessor._process_single_file NDGS Handling: When file_path_obj.suffix.lower() == ".json", ensure it correctly calls self.processor.ndgs_parser.parse_from_file(file_path_obj) and the resulting ndgs_data_for_processor (a JsonDict) is passed to self.processor.process(..., ndgs_input_data_dict=ndgs_data_for_processor).
HTML Report Links: Verify that _generate_html_batch_summary_report creates correct relative links (e.g., ./{safe_job_id}/{safe_job_id}_report.html) to individual job HTML reports, assuming these reports are saved by TextProcessor._finalize_results (via Part 14's implementation) and then copied or managed by _save_individual_job_results.
Priority: Medium.
Dependencies: Part 14 (TextProcessor HTML report generation), Part 15 (NDGSIntegration).
Part 17: Main Execution Function and Entry Point (main, parse_arguments)
Objective: Create a fully functional CLI and robust main execution flow.
Specific Tasks:
parse_arguments Default Value Accuracy: Ensure all default=None arguments that should derive defaults from NGGSConfig (e.g., --perspective, --phase-focus, --colloquial-level, --max-workers) correctly use getattr(temp_config_for_help, 'CONFIG_ATTRIBUTE_NAME', fallback_value) to display accurate defaults in help messages. Match attribute names with those added in Part 1.
setup_components Template Loading: Modify to use load_template(..., config=effective_config) for all templates, removing the _DUMMY_DEFAULT_TEMPLATES and ensuring load_template uses effective_config to find template files or fall back to the global DEFAULT_TEMPLATES (as per load_template's internal logic revised in Part 4).
HTML Report Path Handling: In run_single_job, expect job_result_dict["html_report_path"] (set by TextProcessor._finalize_results in Part 14) to contain the path to the HTML report, not its content. Log this path. The saving of the HTML file itself is TextProcessor's responsibility.
Priority: High (Entry point for all operations).
Dependencies: All other Parts, especially Part 1 (NGGSConfig defaults), Part 14 (TextProcessor's HTML report path).
Part 18: Script Entry Point
Objective: Finalize script startup and shutdown.
Specific Tasks:
CONCURRENT_FUTURES_AVAILABLE Definition: Confirm this flag is correctly defined in Part 1 based on concurrent.futures import attempt.
Logger Initialization: Remove the entry_point_logger. Rely solely on the logger instance that is first configured with basic settings at the start of main() (Part 17) and then fully configured by setup_logging() using NGGSConfig. Ensure signal handlers attempt to use this global logger if available.
JSON_REPAIR_AVAILABLE Flag Usage: Ensure this flag (set in Part 18) is correctly referenced in Evaluator._parse_evaluation_response (Part 8) to conditionally enable json_repair functionality.
Priority: Medium (Mainly cleanup and ensuring consistency).
Dependencies: Part 1 (Imports), Part 3 (Logging setup), Part 8 (Evaluator).
4. 📌 Constraints and Non-Goals
Strict Adherence to Python 3.x: No Python 2 specific code. Utilize modern Python 3 features where appropriate (e.g., f-strings, type hints, pathlib).
No New External Dependencies (unless specified for optional features): Beyond google-generativeai and json_repair (optional), do not introduce new third-party libraries without explicit approval. concurrent.futures is standard.
User Guide is Primary Specification: Functional implementation must align with User Guide descriptions. If discrepancies or ambiguities are found between this SoW and the User Guide, prioritize the User Guide and flag the issue.
Focus on Core Logic, Not UI: This task is for backend script finalization. No GUI development is required.
Performance Optimization: While clean code is expected, extensive performance profiling and micro-optimizations are secondary to correctness and completeness for v1.8, unless specifically instructed for a bottleneck.
Backward Compatibility (with v1.7 data formats): Not a primary goal for v1.8 internal data structures, but NDGS input and GLCAI output formats must adhere to specified (or inferred from User Guide) schemas.
5. 📋 Expected Output Format
Primary Deliverable: A single, fully executable Python script named NGGS-Lite_v1.8_Jules_Implemented.py.
Internal Structure:
Shebang: #!/usr/bin/env python3
Encoding: # -*- coding: utf-8 -*-
Script-level docstring detailing purpose and version.
Clear Part demarcations: # === Part X: Description ===
All imports consolidated in Part 1.
Comprehensive docstrings for all classes, methods, and functions (Google style preferred).
Type hints for all function/method signatures and key variables.
Concise comments (Japanese or English) for complex logic sections.
No External Configuration Files (for core operation in v1.8): All core configurations are managed via NGGSConfig defaults and CLI overrides. Template files and vocabulary files are external.
6. 📚 Illustrative Examples (Clarifications)
Example: _create_subjective_template (Part 14 - Conceptual Detail)
Python

# class TextProcessor:
# def _create_subjective_template(self, subj_res: EvaluationResult, ...) -> Result[str, TemplateError]:
#     instructions = ["# 主観的語りの深化指示:"]
#     if subj_res.components.get("first_person_score", 5.0) < 3.0:
#         instructions.append("- 一人称代名詞の使用頻度や分布を見直し、より没入感を高めるようにしてください。")
#         instructions.append(f"  - 推奨語彙（一人称表現）: {self.vocab_manager.get_vocabulary_for_prompt(tags=['first_person_helper'], count=3)}")
#     # ... other component checks and instruction generation ...
#     return Result.ok("\n".join(instructions))
Example: NGGSConfig Field Addition (Part 1)
Python

# class NGGSConfig:
#     # ... (other fields)
#     GEMINI_SAFETY_SETTINGS: Dict[str, str] = field(default_factory=lambda: {
#         "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", # Example: Block no harassment
#         "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE", # Example
#         # ... other harm categories and block thresholds
#     })
#     LOG_MAX_BYTES_DEFAULT_VAL: int = 10 * 1024 * 1024 # 10MB
#     LOG_BACKUP_COUNT_DEFAULT_VAL: int = 5
7. ✅ Validation and Acceptance Criteria
Static Analysis: Script passes flake8 and mypy (strict mode if possible) without errors or significant warnings.
Compilation: python -m py_compile NGGS-Lite_v1.8_Jules_Implemented.py executes successfully.
Execution:
Script runs without Python exceptions when provided with valid sample inputs (text file, direct text, NDGS JSON, batch directory).
All CLI arguments defined in parse_arguments (Part 17) are functional and correctly influence behavior/configuration.
Functional Verification:
Core text generation produces Gothic-style text.
All evaluation metrics (ETI, RI, Subjective, Derived Scores) are calculated and present in the output JSON.
Iterative improvement loop (if max_loops > 0) functions, generating multiple versions and selecting the best based on overall_quality.
NDGS input (if provided via --ndgs-input) correctly influences initial text and/or parameters.
GLCAI feedback JSON file is generated in the correct format and location for each job.
HTML report (even if basic for v1.8) is generated and path is correctly logged/stored for single jobs. Batch HTML summary is generated.
Output Verification:
JSON output files (_results.json for individual jobs, _batch_summary_report.json) are well-formed and contain all expected data fields.
Final generated text (_final.txt) is saved.
8. 🔄 Feedback and Iteration Protocol
Initial Review: Upon receipt of the implemented script, a preliminary review will be conducted focusing on critical path execution and major unimplemented features based on this SoW.
Feedback Format: Feedback will be provided referencing specific Part numbers and method/class names, detailing discrepancies from this SoW or observed issues.
Example Feedback 1: "Part 14, _create_emotion_template: The generated instruction does not sufficiently address cases where emotion_arc_target is complex (e.g., 'A->B->C->D'). Please add logic to provide guidance for multi-step arcs."
Example Feedback 2: "Part 12, TextProcessor.process: GLCAI track_vocabulary_usage is not called after the initial generation in Loop 0 when skip_initial_generation_flag is False. This needs to be added."
Re-submission: Jules will address all feedback points and resubmit the updated script. This cycle will repeat until all requirements are met and acceptance criteria are passed.
9. 📅 Schedule and Priority
Overall Deadline: 3 weeks from SoW acceptance.
Phase 1 (Week 1 - Critical Path & Core Implementation):
Highest Priority: Part 14 (_create_*_template methods full implementation, _generate_text full implementation, HTML report generation in _finalize_results).
Part 1 (NGGSConfig field additions).
Part 11 & 12 (Activate GLCAI track_vocabulary_usage calls, implement concrete _calculate_emotion_arc_score and _calculate_colloquial_score).
Part 15 (Implement detailed NDGS parameter mapping, ensure GLCAI calls are robust).
Phase 2 (Week 2 - Evaluation Refinement & Loop Finalization):
Part 9 (Refine ETI/RI heuristics, ensure DEFAULT_EVALUATION_TEMPLATE key matching).
Part 13 (Complete _perform_improvement_loop logic including best version selection, implement dedicated LLM instruction generation template).
Part 5, Part 8 (Refine LLMClient error handling, ensure Evaluator uses json_repair conditionally).
Phase 3 (Week 3 - Integrations, Batch, Final Polish & Testing):
Part 10 (Enhance Phase/Layer detection, implement perspective_mode handling in SubjectiveEvaluator).
Part 6 & 7 (Finalize vocabulary style adjustments, ETI category mapping).
Part 16 (Verify BatchProcessor NDGS handling and HTML report links).
Part 17, Part 18 (Finalize CLI, main(), entry point, import cleanup).
Comprehensive testing (Unit & Integration).
10. 🙏 Final Request to AI Agent (Jules)
Your primary directive is to produce a fully executable, robust, and feature-complete Python script (NGGS-Lite_v1.8_Jules_Implemented.py) that strictly adheres to this Statement of Work and all referenced documents (User Guide, Roadmap, Final Check Document).

Focus on:

Implementing all currently stubbed/unimplemented functionalities, especially the _create_*_template methods (Part 14) and the core improvement loop logic within TextProcessor (Part 13).
Ensuring all User Guide features are operational, including all evaluation metrics and external system integrations (NDGS, GLCAI).
Resolving all issues identified in the "Final Check Document."
Achieving zero runtime errors under normal operating conditions with valid inputs.
Maintaining code quality through PEP8 compliance, comprehensive type hinting, and clear documentation.
If ambiguities arise or technical constraints prevent exact adherence to a minor specification, implement the most robust and logical solution, clearly documenting your reasoning and any deviations in your submission report. Your expertise in creating high-quality, production-ready code is paramount.

11. Key Refinement Points in This SoW (v3.0 for jules)
This version of the SoW has been refined from previous internal (Gemini-generated) versions to provide Jules with maximum clarity and actionability:

Directives from "Final Check": All critical unimplemented items and necessary fixes identified in the "NGGS-Lite v1.8 (Refactoring Edition) Final Check and Commentary" have been explicitly translated into specific tasks within the relevant Parts.
Increased Specificity: Instructions for method implementations (e.g., _create_*_template, _generate_text, NDGS parsing, GLCAI tracking) are more detailed regarding expected inputs, outputs, and internal logic.
Emphasis on User Guide Features: Explicit calls to implement all User Guide features, with references to specific sections where appropriate.
Configuration Completeness: Specific new fields for NGGSConfig are listed to ensure all configurable aspects are covered.
Error Handling & Robustness: Continued emphasis on using the Result pattern and custom exceptions, with specific instructions for handling API errors and content blocks.
Clear Deliverables: Defines the expected Python script and supplementary materials.
Structured Schedule: Provides a phased approach to implementation, prioritizing critical path functionalities.
This document should serve as a comprehensive guide for Jules to complete the NGGS-Lite v1.8 project to the required standards.
