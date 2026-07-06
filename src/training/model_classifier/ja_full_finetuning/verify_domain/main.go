// Package main は ja_domain_classifier_full.py で学習した日本語分野分類モデル
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

// ModelConfig は config.json の architectures フィールドを読み取るための構造体。
type ModelConfig struct {
	Architectures []string `json:"architectures"`
}

// CategoryMapping は category_mapping.json（idx_to_category / category_to_idx）を読み取る。
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

var categoryLabels map[int]string

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

func loadCategoryMapping(modelPath string) error {
	mappingPath := filepath.Join(modelPath, "category_mapping.json")
	data, err := os.ReadFile(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to read mapping file %s: %w", mappingPath, err)
	}

	var mapping CategoryMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return fmt.Errorf("failed to parse mapping JSON: %w", err)
	}

	categoryLabels = make(map[int]string)
	for idxStr, label := range mapping.IdxToCategory {
		var idx int
		if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
			return fmt.Errorf("failed to parse category index %s: %w", idxStr, err)
		}
		categoryLabels[idx] = label
	}
	fmt.Printf("Loaded %d category mappings\n", len(categoryLabels))
	return nil
}

func main() {
	var (
		modelPath = flag.String("model", "ja_full_domain_classifier_ruri-v3-30m", "Path to the JA domain classifier model")
		useCPU    = flag.Bool("cpu", true, "Use CPU instead of GPU")
	)
	flag.Parse()

	fmt.Println("日本語 分野分類（JMMLU）モデル検証")
	fmt.Println("=====================================")

	architecture, err := detectModelArchitecture(*modelPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}
	fmt.Printf("Detected model architecture: %s\n", architecture)
	if !strings.Contains(architecture, "ModernBert") {
		log.Fatalf("unsupported model architecture for this verifier: %s (expected ModernBertForSequenceClassification)", architecture)
	}

	if err := loadCategoryMapping(*modelPath); err != nil {
		log.Fatalf("Failed to load category mapping: %v", err)
	}

	if err := candle.InitModernBertClassifier(*modelPath, *useCPU); err != nil {
		log.Fatalf("Failed to initialize ModernBERT classifier: %v", err)
	}
	fmt.Println("Domain classifier initialized successfully!")

	testSamples := []struct {
		text     string
		expected string
	}{
		{"企業合併における最良の戦略は何ですか？", "business"},
		{"独占禁止法は企業間の競争にどのような影響を与えますか？", "business"},
		{"消費者行動に影響を与える心理的要因は何ですか？", "psychology"},
		{"契約成立の法的要件を説明してください。", "law"},
		{"民法と刑法の違いは何ですか？", "law"},
		{"光合成の仕組みを説明してください。", "biology"},
		{"eのx乗の微分は何ですか？", "math"},
		{"需要と供給の経済原則を説明してください。", "economics"},
		{"真核細胞におけるDNA複製の仕組みは？", "biology"},
		{"コンピュータのトランジスタの仕組みを説明してください。", "computer science"},
		{"星が瞬いて見えるのはなぜですか？", "physics"},
		{"ローマ帝国の歴史的意義を説明してください。", "history"},
		{"フランスの首都はどこですか？", "other"},
	}

	fmt.Println("\nTesting Domain Classification:")
	fmt.Println(strings.Repeat("=", 50))

	correct := 0
	for i, test := range testSamples {
		result, err := candle.ClassifyModernBertText(test.text)
		if err != nil {
			fmt.Printf("Test %d: classification failed: %v\n", i+1, err)
			continue
		}

		label := categoryLabels[result.Class]
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
