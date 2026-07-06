package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type IntentInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type IntentInitializerImpl struct{}

func (c *IntentInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	if isModernBertModel(modelID) {
		if err := candle_binding.InitModernBertIntentClassifier(modelID, useCPU); err == nil {
			logging.ComponentEvent("classifier", "intent_classifier_initialized", map[string]interface{}{
				"backend":   "modernbert",
				"model_ref": modelID,
			})
			return nil
		}
	}

	success := candle_binding.InitCandleBertClassifier(modelID, numClasses[0], useCPU)
	if success {
		logging.ComponentEvent("classifier", "intent_classifier_initialized", map[string]interface{}{
			"backend":   "candle_bert_auto",
			"model_ref": modelID,
		})
		return nil
	}

	if err := candle_binding.InitModernBertIntentClassifier(modelID, useCPU); err != nil {
		return fmt.Errorf("failed to initialize intent classifier: %w", err)
	}
	logging.ComponentEvent("classifier", "intent_classifier_initialized", map[string]interface{}{
		"backend":   "modernbert",
		"model_ref": modelID,
	})
	return nil
}

func createIntentInitializer() IntentInitializer {
	return &IntentInitializerImpl{}
}

type IntentInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
	ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error)
}

type IntentInferenceImpl struct{}

func (c *IntentInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	result, err := candle_binding.ClassifyModernBertIntentText(text)
	if err == nil {
		return result, nil
	}
	return candle_binding.ClassifyCandleBertText(text)
}

func (c *IntentInferenceImpl) ClassifyWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	return candle_binding.ClassifyModernBertIntentTextWithProbabilities(text)
}

func createIntentInference() IntentInference {
	return &IntentInferenceImpl{}
}

func withIntent(intentMapping *CategoryMapping, intentInitializer IntentInitializer, intentInference IntentInference) option {
	return func(c *Classifier) {
		c.IntentMapping = intentMapping
		c.intentInitializer = intentInitializer
		c.intentInference = intentInference
	}
}
