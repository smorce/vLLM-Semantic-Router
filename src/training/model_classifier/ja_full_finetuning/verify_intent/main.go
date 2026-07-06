// Package main は ja_intent_classifier_full.py で学習した日本語意図分類モデル
// （cl-nagoya/ruri-v3-30m フルファインチューニング、ModernBertForSequenceClassification）を
// candle-binding 経由で検証する。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
)

type ModelConfig struct {
	Architectures []string `json:"architectures"`
}

// IntentMapping は label_mapping.json（label_to_id / id_to_label）を読み取る。
type IntentMapping struct {
	LabelToID map[string]int    `json:"label_to_id"`
	IDToLabel map[string]string `json:"id_to_label"`
}

var intentLabels map[int]string

func detectModelArchitecture(modelPath string) (string, error) {
	configPath := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return "", fmt.Errorf("failed to read config.json: %w", err)
	}

	var config ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return "", fmt.Errorf("failed to parse config.json: %w", err)
	}
	if len(config.Architectures) == 0 {
		return "", fmt.Errorf("no architectures found in config.json")
	}
	return config.Architectures[0], nil
}

func loadIntentMapping(modelPath string) error {
	mappingPath := filepath.Join(modelPath, "label_mapping.json")
	data, err := os.ReadFile(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to read intent mapping file %s: %w", mappingPath, err)
	}

	var mapping IntentMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return fmt.Errorf("failed to parse intent mapping JSON: %w", err)
	}

	intentLabels = make(map[int]string)
	for idxStr, label := range mapping.IDToLabel {
		var idx int
		if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
			return fmt.Errorf("failed to parse label index %s: %w", idxStr, err)
		}
		intentLabels[idx] = label
	}
	fmt.Printf("Loaded %d intent label mappings\n", len(intentLabels))
	return nil
}

func main() {
	var (
		modelPath = flag.String("model", "ja_full_intent_classifier_ruri-v3-30m", "Path to the JA intent classifier model")
		useCPU    = flag.Bool("cpu", true, "Use CPU instead of GPU")
	)
	flag.Parse()

	fmt.Println("日本語 意図分類（8カテゴリ）モデル検証")
	fmt.Println("=========================================")

	architecture, err := detectModelArchitecture(*modelPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}
	fmt.Printf("Detected model architecture: %s\n", architecture)
	if !strings.Contains(architecture, "ModernBert") {
		log.Fatalf("unsupported model architecture for this verifier: %s (expected ModernBertForSequenceClassification)", architecture)
	}

	if err := loadIntentMapping(*modelPath); err != nil {
		log.Fatalf("Failed to load intent mapping: %v", err)
	}

	// 意図分類は本番 router の classifier.domain/classifier.pii のような専用スロットを
	// 持たないため、汎用の ModernBERT テキスト分類器として検証する。
	if err := candle.InitModernBertClassifier(*modelPath, *useCPU); err != nil {
		log.Fatalf("Failed to initialize ModernBERT classifier: %v", err)
	}
	fmt.Println("Intent classifier initialized successfully!")

	testSamples := []struct {
		text     string
		expected string
	}{
		{"東京の明日の天気を教えてください。", "information_retrieval"},
		{"100ドルを日本円に換算するといくらですか？", "calculation"},
		{"来週の月曜日にチームミーティングの予定を入れてください。", "scheduling"},
		{"山田さんにお礼のメールを送ってください。", "communication"},
		{"新しいメモを作成して、買い物リストを保存してください。", "file_operations"},
		{"パスワードをランダムに生成してください。", "data_transformation"},
		{"このレビューの感情を分析してください。", "analysis"},
		{"こんにちは、調子はどうですか？", "no_function_needed"},
	}

	fmt.Println("\nTesting Intent Classification:")
	fmt.Println(strings.Repeat("=", 50))

	correct := 0
	for i, test := range testSamples {
		result, err := candle.ClassifyModernBertText(test.text)
		if err != nil {
			fmt.Printf("Test %d: classification failed: %v\n", i+1, err)
			continue
		}

		label := intentLabels[result.Class]
		if label == "" {
			label = fmt.Sprintf("Class_%d", result.Class)
		}

		status := "✗ INCORRECT"
		if label == test.expected {
			status = "CORRECT"
			correct++
		}
		fmt.Printf("Test %d: \"%s\"\n  -> %s (expected: %s, confidence: %.4f) %s\n",
			i+1, test.text, label, test.expected, result.Confidence, status)
	}

	total := len(testSamples)
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("SUMMARY: %d/%d correct (%.1f%%)\n", correct, total, float64(correct)/float64(total)*100)
}
