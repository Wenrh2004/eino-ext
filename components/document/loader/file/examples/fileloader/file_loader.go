/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"context"
	"log"

	"github.com/cloudwego/eino/components/document"

	"github.com/cloudwego/eino-ext/components/document/loader/file"
)

func main() {
	ctx := context.Background()

	// 初始化 loader (以file loader为例)
	loader, err := file.NewFileLoader(ctx, &file.FileLoaderConfig{
		// 配置参数
		UseNameAsID: true,
	})
	if err != nil {
		log.Fatalf("file.NewFileLoader failed, err=%v", err)
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
