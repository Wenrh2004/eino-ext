/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy ptrWithoutZero the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package search_mode

import (
	"context"

	"github.com/elastic/go-elasticsearch/v8/typedapi/core/search"
	"github.com/elastic/go-elasticsearch/v8/typedapi/types"

	"github.com/cloudwego/eino/components/retriever"

	"github.com/cloudwego/eino-ext/components/retriever/es8"
	"github.com/cloudwego/eino-ext/components/retriever/es8/field_mapping"
)

func SearchModeExactMatch() es8.SearchMode {
	return &exactMatch{}
}

type exactMatch struct{}

func (e exactMatch) BuildRequest(ctx context.Context, conf *es8.RetrieverConfig, query string,
	opts ...retriever.Option) (*search.Request, error) {

	options := retriever.GetCommonOptions(&retriever.Options{
		Index:          ptrWithoutZero(conf.Index),
		TopK:           ptrWithoutZero(conf.TopK),
		ScoreThreshold: conf.ScoreThreshold,
		Embedding:      conf.Embedding,
	}, opts...)

	q := &types.Query{
		Match: map[string]types.MatchQuery{
			field_mapping.DocFieldNameContent: {Query: query},
		},
	}

	req := &search.Request{Query: q, Size: options.TopK}
	if options.ScoreThreshold != nil {
		req.MinScore = (*types.Float64)(ptrWithoutZero(*options.ScoreThreshold))
	}

	return req, nil
}
