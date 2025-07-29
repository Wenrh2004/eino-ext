package milvus

import (
	"fmt"
	
	"github.com/cloudwego/eino/components/embedding"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type RetrieverConfig struct {
	Client *milvusclient.Client
	
	Collection string
	TopK       int
	
	Embedding         embedding.Embedder
	DocumentConverter DocumentConverter
	VectorConverter   VectorConverter
}

func (c *RetrieverConfig) validate() error {
	if c.Client == nil {
		return fmt.Errorf("[Retriever.RetrieverConfig] milvus client is nil")
	}
	if c.Collection == "" {
		c.Collection = defaultCollection
	}
	if c.TopK <= 0 {
		c.TopK = defaultTopK
	}
	if c.DocumentConverter == nil {
		c.DocumentConverter = getDefaultDocumentConverter()
	}
	if c.VectorConverter == nil {
		c.VectorConverter = getDefaultVectorConverter()
	}
	return nil
}
