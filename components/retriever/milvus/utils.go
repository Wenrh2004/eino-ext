package milvus

import (
	"context"
	
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/embedding"
)

func makeEmbeddingCtx(ctx context.Context, emb embedding.Embedder) context.Context {
	runInfo := &callbacks.RunInfo{
		Component: components.ComponentOfEmbedding,
	}
	
	if embType, ok := components.GetType(emb); ok {
		runInfo.Type = embType
	}
	
	runInfo.Name = runInfo.Type + string(runInfo.Component)
	
	return callbacks.ReuseHandlers(ctx, runInfo)
}

func vector2Float32(vector []float64) []float32 {
	float32Arr := make([]float32, len(vector))
	for i, v := range vector {
		float32Arr[i] = float32(v)
	}
	return float32Arr
}
