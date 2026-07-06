package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSignalReadinessPreferenceNotReadyWhenClassifierNil(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					PreferenceRules: []config.PreferenceRule{{Name: "code_generation"}},
				},
			},
			InlineModels: config.InlineModels{
				Classifier: config.Classifier{
					PreferenceModel: config.PreferenceModelConfig{
						UseContrastive: prefBoolPtr(true),
					},
				},
			},
		},
	}

	ready := classifier.signalReadiness()
	if ready[config.SignalTypePreference] {
		t.Fatal("expected preference signal to be not ready when classifier is nil")
	}
}

func TestEvaluatePreferenceSignalNilClassifierDoesNotPanic(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					PreferenceRules: []config.PreferenceRule{{Name: "code_generation"}},
				},
			},
		},
	}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}}

	classifier.evaluatePreferenceSignal(results, nil, "hello")
}

func TestPreferenceClassifierClassifyNilReceiver(t *testing.T) {
	var classifier *PreferenceClassifier
	result, err := classifier.Classify(`[{"role":"user","content":"hello"}]`)
	if err == nil {
		t.Fatal("expected error for nil preference classifier")
	}
	if result != nil {
		t.Fatalf("expected nil result, got %+v", result)
	}
}
