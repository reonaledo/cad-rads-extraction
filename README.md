The manuscript has been submitted to Korean Journal of Radiology.
File structure will be reorganized for better accessibility soon.

# Large Language Models for CAD-RADS 2.0 Extraction from Semi-Structured Coronary CT Angiography Reports: A Multi-Institutional Study

## Abstract
Objective: To evaluate the accuracy of large language models (LLMs) in extracting Coronary Artery Disease Reporting and Data System (CAD-RADS) 2.0 components from coronary CT angiography (CCTA) reports and assess the impact of prompting strategies.

Materials and Methods: In this multi-institutional study, we collected 319 synthetic semi-structured CCTA reports from six institutions to protect patient privacy while maintaining clinical relevance. The dataset included 150 reports from the primary institution (100 for instruction development, 50 for internal testing) and 169 reports from five external institutions for external testing. Board-certified radiologists established reference standards following CAD-RADS 2.0 guidelines for all three components: stenosis severity, plaque burden, and modifiers. Six LLMs (GPT-4, GPT-4o, Claude-3.5-Sonnet, o1-mini, Gemini-1.5-Pro, and DeepSeek-R1-Distill-Qwen-14B) were evaluated using an optimized instruction with prompting strategies, including zero-shot or few-shot with or without chain-of-thought (CoT) prompting. Accuracy was assessed and compared with the McNemar’s test.

Results: LLMs demonstrated robust accuracy across all CAD-RADS 2.0 components. Peak stenosis severity accuracies reached 0.980 (48/49, Claude-3.5-Sonnet and o1-mini) in internal testing and 0.946 (158/167, GPT-4o and o1-mini) in external testing. Plaque burden extraction showed exceptional accuracy, with multiple models achieving perfect accuracy (43/43) in internal testing and up to 0.993 (137/138, GPT-4o and o1-mini) in external testing. Modifier detection demonstrated consistently high accuracy (≥0.990) across most models. The one open-source model, DeepSeek-R1-Distill-Qwen-14B, showed relatively lower accuracy for stenosis accuracies: 0.898 (44/49, internal) and 0.820 (137/167, external). CoT prompting significantly enhanced accuracy for several models, with GPT-4 showing the most substantial improvements: stenosis severity accuracy increased by 0.192 (p<0.001) and plaque burden accuracy by 0.152 (p<0.001) in external testing. 

Conclusions: LLMs demonstrated high accuracy in automated extraction of CAD-RADS 2.0 components from semi-structured CCTA reports, particularly when used with CoT prompting.

Keywords
Coronary CT Angiography, CAD-RADS 2.0, Information Extraction, Large Language Model, Prompting Strategy  
