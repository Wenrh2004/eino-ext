package main

import (
	"context"
	"log"
	"time"

	"github.com/cloudwego/eino/components/document"
)

func main() {
	ctx := context.Background()

	log.Printf("===== call Custom Loader directly =====")
	// 初始化 loader
	loader, err := NewCustomLoader(&Config{
		DefaultTimeout:    10 * time.Second,
		DefaultRetryCount: 10,
	})
	if err != nil {
		log.Fatalf("NewCustomLoader failed, err=%v", err)
	}

	// 加载文档
	filePath := "../../testdata/test.md"
	docs, err := loader.Load(ctx, document.Source{
		URI: filePath,
	})
	if err != nil {
		log.Fatalf("loader.Load failed, err=%v", err)
	}

	log.Printf("doc content: %v", docs[0].Content)
}
