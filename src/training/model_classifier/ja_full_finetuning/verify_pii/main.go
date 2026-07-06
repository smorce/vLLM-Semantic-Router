// Package main は ja_pii_full.py で学習した日本語PII検出モデル
// （cl-nagoya/ruri-v3-30m フルファインチューニング、ModernBertForTokenClassification）を
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

// combineBIOEntities は BIO タグ付きトークンを完全なエンティティへ結合する。
func combineBIOEntities(rawEntities []candle.TokenEntity, originalText string) []candle.TokenEntity {
	if len(rawEntities) == 0 {
		return rawEntities
	}

	var combined []candle.TokenEntity
	var current *candle.TokenEntity

	for _, entity := range rawEntities {
		entityType := entity.EntityType

		switch {
		case strings.HasPrefix(entityType, "B-"):
			if current != nil {
				combined = append(combined, *current)
			}
			baseType := entityType[2:]
			current = &candle.TokenEntity{
				EntityType: baseType,
				Start:      entity.Start,
				End:        entity.End,
				Text:       entity.Text,
				Confidence: entity.Confidence,
			}
		case strings.HasPrefix(entityType, "I-"):
			baseType := entityType[2:]
			if current != nil && current.EntityType == baseType {
				current.End = entity.End
				if current.Start >= 0 && current.End <= len(originalText) && current.Start < current.End {
					current.Text = originalText[current.Start:current.End]
				}
				if entity.Confidence < current.Confidence {
					current.Confidence = entity.Confidence
				}
			} else {
				if current != nil {
					combined = append(combined, *current)
				}
				current = nil
			}
		default:
			if current != nil {
				combined = append(combined, *current)
				current = nil
			}
			if entityType != "O" && entityType != "" {
				combined = append(combined, entity)
			}
		}
	}

	if current != nil {
		combined = append(combined, *current)
	}
	return combined
}

func main() {
	var (
		modelPath = flag.String("model", "ja_full_pii_detector_ruri-v3-30m", "Path to the JA PII token classifier model")
		useCPU    = flag.Bool("cpu", true, "Use CPU instead of GPU")
	)
	flag.Parse()

	fmt.Println("日本語 PII検出モデル検証")
	fmt.Println("============================")

	architecture, err := detectModelArchitecture(*modelPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}
	fmt.Printf("Detected model architecture: %s\n", architecture)
	if !strings.Contains(architecture, "ModernBert") {
		log.Fatalf("unsupported model architecture for this verifier: %s (expected ModernBertForTokenClassification)", architecture)
	}

	if err := candle.InitModernBertPIITokenClassifier(*modelPath, *useCPU); err != nil {
		log.Fatalf("Failed to initialize ModernBERT PII token classifier: %v", err)
	}
	fmt.Println("PII token classifier initialized successfully!")

	testCases := []struct {
		text          string
		description   string
		expectedPII   bool
		expectedTypes []string
	}{
		{
			text:          "山田太郎（yamada@example.co.jp）から請求書番号 INV-2024-0042 で見積書が届いた。",
			description:   "氏名・メール・取引ID検出",
			expectedPII:   true,
			expectedTypes: []string{"HUMAN_NAME", "EMAIL_ADDRESS", "TRANSACTION_ID"},
		},
		{
			text:          "東京都千代田区丸の内1-1-1にお住まいの佐藤花子様、電話番号は03-1234-5678です。",
			description:   "住所・氏名・電話番号検出",
			expectedPII:   true,
			expectedTypes: []string{"ADDRESS", "HUMAN_NAME", "PHONE_NUMBER"},
		},
		{
			text:          "株式会社サンプル商事の担当者にご確認ください。",
			description:   "会社名検出",
			expectedPII:   true,
			expectedTypes: []string{"COMPANY_NAME"},
		},
		{
			text:          "これは個人情報を含まない普通の文章です。",
			description:   "PIIなし",
			expectedPII:   false,
			expectedTypes: []string{},
		},
	}

	fmt.Println("\nTesting PII Detection:")
	fmt.Println(strings.Repeat("=", 60))

	configPath := filepath.Join(*modelPath, "config.json")
	correct := 0
	for i, test := range testCases {
		fmt.Printf("\nTest %d: %s\n", i+1, test.description)
		fmt.Printf("Text: \"%s\"\n", test.text)

		tokenResult, err := candle.ClassifyModernBertPIITokens(test.text, configPath)
		if err != nil {
			fmt.Printf("Classification failed: %v\n", err)
			continue
		}

		tokenResult.Entities = combineBIOEntities(tokenResult.Entities, test.text)

		detectedTypes := make(map[string]bool)
		hasPII := false
		for _, entity := range tokenResult.Entities {
			if entity.Confidence >= 0.5 {
				detectedTypes[strings.ToUpper(entity.EntityType)] = true
				hasPII = true
			}
		}

		var typesList []string
		for t := range detectedTypes {
			typesList = append(typesList, t)
		}

		fmt.Printf("Has PII: %v\n", hasPII)
		if len(typesList) > 0 {
			fmt.Printf("Detected Types: %v\n", typesList)
		}

		predictionCorrect := hasPII == test.expectedPII
		if predictionCorrect {
			fmt.Println("CORRECT")
			correct++
		} else {
			fmt.Printf("✗ INCORRECT (expected HasPII=%v)\n", test.expectedPII)
		}
	}

	total := len(testCases)
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("SUMMARY: %d/%d correct (%.1f%%)\n", correct, total, float64(correct)/float64(total)*100)
}
