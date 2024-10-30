package openai

//
//type EmbeddingConfig struct {
//	// if you want to use Azure OpenAI Service, set the next three fields. refs: https://learn.microsoft.com/en-us/azure/ai-services/openai/
//	// ByAzure set this field to true when using Azure OpenAI Service, otherwise it does not need to be set.
//	ByAzure bool `json:"by_azure"`
//	// BaseURL https://{{$YOUR_RESOURCE_NAME}}.openai.azure.com, YOUR_RESOURCE_NAME is the name of your resource that you have created on Azure.
//	BaseURL string `json:"base_url"`
//	// APIVersion specifies the API version you want to use.
//	APIVersion string `json:"api_version"`
//
//	// APIKey is typically OPENAI_API_KEY, but if you have set up Azure, then it is Azure API_KEY.
//	APIKey string `json:"api_key"`
//
//	// Timeout specifies the http request timeout.
//	Timeout time.Duration `json:"timeout"`
//
//	// The following fields have the same meaning as the fields in the openai embedding API request. Ref: https://platform.openai.com/docs/api-reference/embeddings/create
//	Model          string                            `json:"model"`
//	EncodingFormat *protocol.EmbeddingEncodingFormat `json:"encoding_format,omitempty"`
//	Dimensions     *int                              `json:"dimensions,omitempty"`
//	User           *string                           `json:"user,omitempty"`
//}
//
//var _ embedding.Embedder = (*Embedder)(nil)
//
//type Embedder struct {
//	cli *protocol.OpenAIClient
//}
//
//func NewEmbedder(ctx context.Context, config *EmbeddingConfig) (*Embedder, error) {
//	if config == nil {
//		config = &EmbeddingConfig{Model: string(openai.AdaEmbeddingV2)}
//	}
//
//	cli, err := protocol.NewOpenAIClient(ctx, &protocol.OpenAIConfig{
//		ByAzure:        config.ByAzure,
//		BaseURL:        config.BaseURL,
//		APIVersion:     config.APIVersion,
//		APIKey:         config.APIKey,
//		HTTPClient:     &http.Client{Timeout: config.Timeout},
//		Model:          config.Model,
//		EncodingFormat: config.EncodingFormat,
//		Dimensions:     config.Dimensions,
//		User:           config.User,
//	})
//	if err != nil {
//		return nil, err
//	}
//
//	return &Embedder{
//		cli: cli,
//	}, nil
//}
//
//func (e *Embedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) (
//	embeddings [][]float64, err error) {
//	return e.cli.EmbedStrings(ctx, texts, opts...)
//}
//
//const typ = "OpenAI"
//
//func (e *Embedder) GetType() string {
//	return typ
//}
//
//func (e *Embedder) IsCallbacksEnabled() bool {
//	return true
//}

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/sashabaranov/go-openai"

	"code.byted.org/flow/eino/callbacks"
	"code.byted.org/flow/eino/components/embedding"
)

type EmbeddingEncodingFormat string

const (
	EmbeddingEncodingFormatFloat  EmbeddingEncodingFormat = "float"
	EmbeddingEncodingFormatBase64 EmbeddingEncodingFormat = "base64"
)

type EmbeddingConfig struct {
	// if you want to use Azure OpenAI Service, set the next three fields. refs: https://learn.microsoft.com/en-us/azure/ai-services/openai/
	// ByAzure set this field to true when using Azure OpenAI Service, otherwise it does not need to be set.
	ByAzure bool `json:"by_azure"`
	// BaseURL https://{{$YOUR_RESOURCE_NAME}}.openai.azure.com, YOUR_RESOURCE_NAME is the name of your resource that you have created on Azure.
	BaseURL string `json:"base_url"`
	// APIVersion specifies the API version you want to use.
	APIVersion string `json:"api_version"`

	// APIKey is typically OPENAI_API_KEY, but if you have set up Azure, then it is Azure API_KEY.
	APIKey string `json:"api_key"`

	// Timeout specifies the http request timeout.
	Timeout time.Duration `json:"timeout"`

	// The following fields have the same meaning as the fields in the openai embedding API request. Ref: https://platform.openai.com/docs/api-reference/embeddings/create
	Model          string                   `json:"model"`
	EncodingFormat *EmbeddingEncodingFormat `json:"encoding_format,omitempty"`
	Dimensions     *int                     `json:"dimensions,omitempty"`
	User           *string                  `json:"user,omitempty"`
}

var _ embedding.Embedder = (*Embedder)(nil)

type Embedder struct {
	cli    *openai.Client
	config *EmbeddingConfig
}

func NewEmbedder(ctx context.Context, config *EmbeddingConfig) (*Embedder, error) {
	if config == nil {
		config = &EmbeddingConfig{Model: string(openai.AdaEmbeddingV2)}
	}

	var clientConf openai.ClientConfig

	if config.ByAzure {
		clientConf = openai.DefaultAzureConfig(config.APIKey, config.BaseURL)
	} else {
		clientConf = openai.DefaultConfig(config.APIKey)
	}

	clientConf.HTTPClient = &http.Client{
		Timeout: config.Timeout,
	}

	return &Embedder{
		cli:    openai.NewClientWithConfig(clientConf),
		config: config,
	}, nil
}

func (e *Embedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) (
	embeddings [][]float64, err error) {

	var (
		cbm, cbmOK = callbacks.ManagerFromCtx(ctx)
	)

	defer func() {
		if err != nil && cbmOK {
			_ = cbm.OnError(ctx, err)
		}
	}()

	options := &embedding.Options{
		Model: &e.config.Model,
	}
	options = embedding.GetCommonOptions(options, opts...)

	if options.Model == nil || len(*options.Model) == 0 {
		return nil, fmt.Errorf("open embedder uses empty model")
	}

	req := &openai.EmbeddingRequest{
		Input:          texts,
		Model:          openai.EmbeddingModel(*options.Model),
		User:           dereferenceOrZero(e.config.User),
		EncodingFormat: openai.EmbeddingEncodingFormat(dereferenceOrDefault(e.config.EncodingFormat, EmbeddingEncodingFormatFloat)),
		Dimensions:     dereferenceOrZero(e.config.Dimensions),
	}

	conf := &embedding.Config{
		Model:          string(req.Model),
		EncodingFormat: string(req.EncodingFormat),
	}

	ctx = cbm.OnStart(ctx, &embedding.CallbackInput{
		Texts:  texts,
		Config: conf,
	})

	resp, err := e.cli.CreateEmbeddings(ctx, *req)
	if err != nil {
		return nil, err
	}

	embeddings = make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		res := make([]float64, len(d.Embedding))
		for j, emb := range d.Embedding {
			res[j] = float64(emb)
		}
		embeddings[i] = res
	}

	usage := &embedding.TokenUsage{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
	}

	if cbmOK {
		_ = cbm.OnEnd(ctx, &embedding.CallbackOutput{
			Embeddings: embeddings,
			Config:     conf,
			TokenUsage: usage,
		})
	}

	return embeddings, nil
}

const typ = "OpenAI"

func (e *Embedder) GetType() string {
	return typ
}

func (e *Embedder) IsCallbacksEnabled() bool {
	return true
}
