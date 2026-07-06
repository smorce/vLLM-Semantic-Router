package classification

import (
	"fmt"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// IsIntentEnabled checks if function-call intent classification is properly configured.
func (c *Classifier) IsIntentEnabled() bool {
	return c.Config.IntentModel.ModelID != "" &&
		c.Config.IntentModel.CategoryMappingPath != "" &&
		c.IntentMapping != nil
}

// initializeIntentClassifier initializes the function-call intent classification model.
func (c *Classifier) initializeIntentClassifier() error {
	if !c.IsIntentEnabled() || c.intentInitializer == nil {
		return fmt.Errorf("intent classification is not properly configured")
	}

	numClasses := c.IntentMapping.GetCategoryCount()
	if numClasses < 2 {
		return fmt.Errorf("not enough intent categories for classification, need at least 2, got %d", numClasses)
	}

	logging.ComponentEvent("classifier", "intent_classifier_init_started", map[string]interface{}{
		"model_ref": c.Config.IntentModel.ModelID,
		"classes":   numClasses,
		"use_cpu":   c.Config.IntentModel.UseCPU,
	})

	return c.intentInitializer.Init(
		c.Config.IntentModel.ModelID,
		c.Config.IntentModel.UseCPU,
		numClasses,
	)
}

func (c *Classifier) matchIntentCategories(
	intentResult candle_binding.ClassResultWithProbs,
	topCategoryName string,
) []entropy.CategoryProbability {
	threshold := c.Config.IntentModel.Threshold
	topMatch := intentResult.Confidence >= threshold && topCategoryName != ""

	if len(intentResult.Probabilities) == 0 {
		if topMatch {
			return []entropy.CategoryProbability{
				{Category: topCategoryName, Probability: intentResult.Confidence},
			}
		}
		return nil
	}

	entropyResult := entropy.AnalyzeEntropy(intentResult.Probabilities)
	logging.Debugf("[Signal Computation] Intent entropy analysis: entropy=%.3f, normalized=%.3f, uncertainty=%s",
		entropyResult.Entropy, entropyResult.NormalizedEntropy, entropyResult.UncertaintyLevel)

	categoryNames := make([]string, len(intentResult.Probabilities))
	for i := range intentResult.Probabilities {
		if name, ok := c.IntentMapping.GetCategoryFromIndex(i); ok {
			categoryNames[i] = name
		}
	}

	var matched []entropy.CategoryProbability
	switch entropyResult.UncertaintyLevel {
	case "very_low", "low":
		if topMatch {
			matched = []entropy.CategoryProbability{
				{Category: topCategoryName, Probability: intentResult.Confidence},
			}
		}
	default:
		for i, prob := range intentResult.Probabilities {
			if prob >= threshold && categoryNames[i] != "" {
				matched = append(matched, entropy.CategoryProbability{
					Category:    categoryNames[i],
					Probability: prob,
				})
			}
		}
	}

	logging.Debugf("[Signal Computation] Intent signal matched %d categories (uncertainty=%s)",
		len(matched), entropyResult.UncertaintyLevel)
	return matched
}

func (c *Classifier) evaluateIntentSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	intentResult, err := c.intentInference.ClassifyWithProbabilities(text)
	if err != nil {
		logging.Debugf("[Signal Computation] Intent ClassifyWithProbabilities unavailable, falling back to Classify: %v", err)
		basicResult, basicErr := c.intentInference.Classify(text)
		if basicErr != nil {
			err = basicErr
		} else {
			intentResult = candle_binding.ClassResultWithProbs{
				Class:      basicResult.Class,
				Confidence: basicResult.Confidence,
			}
			err = nil
		}
	}
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	categoryName := ""
	if err == nil {
		if name, ok := c.IntentMapping.GetCategoryFromIndex(intentResult.Class); ok {
			categoryName = name
		}
	}

	metrics.RecordSignalExtraction(config.SignalTypeIntent, categoryName, latencySeconds)

	results.Metrics.Intent.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if categoryName != "" && err == nil {
		results.Metrics.Intent.Confidence = float64(intentResult.Confidence)
	}
	logging.Debugf("[Signal Computation] Intent signal evaluation completed in %v", elapsed)

	if err != nil {
		logging.Errorf("intent rule evaluation failed: %v", err)
		return
	}

	matched := c.matchIntentCategories(intentResult, categoryName)
	mu.Lock()
	defer mu.Unlock()
	for _, cat := range matched {
		metrics.RecordSignalMatch(config.SignalTypeIntent, cat.Category)
		results.MatchedIntentRules = append(results.MatchedIntentRules, cat.Category)
		results.SignalConfidences["intent:"+cat.Category] = float64(cat.Probability)
	}
}
