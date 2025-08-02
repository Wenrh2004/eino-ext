package milvus

import (
	"context"
	"fmt"
	"testing"

	. "github.com/bytedance/mockey"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/smartystreets/goconvey/convey"
	"github.com/stretchr/testify/mock"
)

// 模拟Embedding实现
type mockEmbedding struct {
	vectors [][]float64
	err     error
}

func (m *mockEmbedding) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.vectors != nil {
		return m.vectors, nil
	}
	// 默认返回
	result := make([][]float64, len(texts))
	for i := range texts {
		result[i] = []float64{0.1, 0.2, 0.3, 0.4}
	}
	return result, nil
}

// MockMilvusClient 是 MilvusClient 的 mock 实现
type MockMilvusClient struct {
	mock.Mock
}

func TestIndexer_Store(t *testing.T) {
	PatchConvey("test GetCommonOptions", t, func() {
		mockemb := &mockEmbedding{
			vectors: [][]float64{
				{0.1, 0.2, 0.3, 0.4},
				{0.5, 0.6, 0.7, 0.8},
			},
			err: nil,
		}
		mockMilvusClient := &milvusclient.Client{}
		Mock(GetMethod(mockMilvusClient, "HasCollection")).Return(true, nil).Build()
		ctx := context.Background()
		// 创建测试文档
		docs := []*schema.Document{
			{
				ID:       "doc1",
				Content:  "This is a test document",
				MetaData: map[string]interface{}{"key1": "value1"},
			},
			{
				ID:       "doc2",
				Content:  "This is another test document",
				MetaData: map[string]interface{}{"key2": "value2"},
			},
		}

		PatchConvey("test with nil SubIndexes", func() {
			mockco := indexer.GetCommonOptions(&indexer.Options{
				SubIndexes: nil,
				Embedding:  mockemb,
			})

			convey.So(mockco, convey.ShouldNotBeNil)
			convey.So(mockco.SubIndexes, convey.ShouldBeNil)
			convey.So(mockco.Embedding, convey.ShouldEqual, mockemb)
		})

		PatchConvey("test with custom embedding option", func() {
			opts := []indexer.Option{
				indexer.WithEmbedding(mockemb),
			}

			mockco := indexer.GetCommonOptions(&indexer.Options{
				SubIndexes: nil,
				Embedding:  mockemb,
			}, opts...)

			convey.So(mockco, convey.ShouldNotBeNil)
			convey.So(mockco.Embedding, convey.ShouldEqual, mockemb) // 应该被选项覆盖
		})
		PatchConvey("test GetImplSpecificOptions with empty base", func() {
			opts := []indexer.Option{
				indexer.WithEmbedding(mockemb),
			}
			mockio := indexer.GetImplSpecificOptions(&ImplOptions{}, opts...)
			convey.So(mockio.Upsert, convey.ShouldEqual, false)
			convey.So(mockio.Partition, convey.ShouldEqual, "")

		})
		PatchConvey("test GetImplSpecificOptions with base", func() {
			opts := []indexer.Option{
				indexer.WithEmbedding(mockemb),
			}
			mockio := indexer.GetImplSpecificOptions(&ImplOptions{
				Upsert:    true,
				Partition: "test",
			}, opts...)
			convey.So(mockio.Upsert, convey.ShouldEqual, true)
			convey.So(mockio.Partition, convey.ShouldEqual, "test")
		})
		PatchConvey("test embedding error handling", func() {
			// 创建一个会返回错误的mock embedding
			mockEmbWithError := &mockEmbedding{
				err: fmt.Errorf("embedding service unavailable"),
			}

			texts := make([]string, 0, len(docs))
			for _, doc := range docs {
				texts = append(texts, doc.Content)
			}

			// 测试有错误的情况
			mockvetor, err := mockEmbWithError.EmbedStrings(nil, texts)
			convey.So(err.Error(), convey.ShouldContainSubstring, "embedding service unavailable")
			convey.So(mockvetor, convey.ShouldBeNil)
		})
		PatchConvey("test EmbeddingStrings success", func() {
			opts := []indexer.Option{
				indexer.WithEmbedding(mockemb),
			}
			mockco := indexer.GetCommonOptions(&indexer.Options{
				SubIndexes: []string{"test"},
				Embedding:  mockemb,
			}, opts...)
			emb := mockco.Embedding
			texts := make([]string, 0, len(docs))
			for _, doc := range docs {
				texts = append(texts, doc.Content)
			}
			mockvetor, err := emb.EmbedStrings(makeEmbeddingCtx(ctx, emb), texts)
			convey.So(mockvetor, convey.ShouldNotBeEmpty)
			convey.So(err, convey.ShouldBeNil)
		})
		PatchConvey("test vector and document count mismatch", func() {
			// 创建一个返回向量数与文档数不匹配的mock embedding
			mockEmbWithMismatch := &mockEmbedding{
				vectors: [][]float64{{0.1, 0.2, 0.3, 0.4}}, // 只返回一个向量
			}

			texts := make([]string, 0, len(docs))
			for _, doc := range docs {
				texts = append(texts, doc.Content)
			}

			// 调用EmbedStrings获取向量
			mockvetor, err := mockEmbWithMismatch.EmbedStrings(nil, texts)
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockvetor, convey.ShouldNotBeNil)
			// 验证向量数量与文档数量不匹配
			convey.So(len(mockvetor), convey.ShouldNotEqual, len(docs))
		})

		PatchConvey("test DocumentConverter error handling", func() {
			// 创建一个会返回错误的mock converter
			mockConverterWithError := func(_ []*schema.Document, _ int, _ [][]float64) ([]column.Column, error) {
				return nil, fmt.Errorf("document converter error")
			}

			// 测试converter返回错误的情况
			_, err := mockConverterWithError(docs, 4, [][]float64{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}})
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "document converter error")
		})

		PatchConvey("test successful DocumentConverter call", func() {
			// 使用默认converter测试正常情况
			rows, err := defaultDocumentConverter(docs, 4, [][]float64{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}})
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(rows), convey.ShouldEqual, 4)
		})

		PatchConvey("test Upsert option", func() {
			opts := []indexer.Option{
				WithUpsert(),
			}
			mockio := indexer.GetImplSpecificOptions(&ImplOptions{}, opts...)
			convey.So(mockio.Upsert, convey.ShouldBeTrue)
		})

		PatchConvey("test Partition option", func() {
			testPartition := "test_partition"
			opts := []indexer.Option{
				WithPartition(testPartition),
			}
			mockio := indexer.GetImplSpecificOptions(&ImplOptions{}, opts...)
			convey.So(mockio.Partition, convey.ShouldEqual, testPartition)
		})
		PatchConvey("test store success", func() {
			mockIDs := column.NewColumnVarChar("id", []string{"doc1", "doc2"})
			Mock(mockMilvusClient.Insert).Return(mockIDs, nil).Build()
			Mock((*milvusclient.Client).Upsert).Return(milvusclient.UpsertResult{
				UpsertCount: 2,
				IDs:         mockIDs,
			}, nil).Build()
			// 创建索引器
			mockConf := &IndexerConfig{
				Client:     mockMilvusClient,
				Collection: defaultCollection,
				Embedding:  mockemb,
				Dim:        4,
			}

			indexer, err := NewIndexer(ctx, mockConf)

			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)

			// 测试成功存储的情况
			ids, err := indexer.Store(ctx, docs, WithUpsert())
			convey.So(err, convey.ShouldBeNil)
			convey.So(ids, convey.ShouldNotBeNil)
			convey.So(len(ids), convey.ShouldEqual, 2)
			convey.So(ids[0], convey.ShouldEqual, "doc1")
			convey.So(ids[1], convey.ShouldEqual, "doc2")
		})
		PatchConvey("test with upsert error", func() {
			//模拟Upsert失败的情况
			Mock((*milvusclient.Client).Upsert).Return(milvusclient.UpsertResult{}, fmt.Errorf("upsert failed")).Build()
			mockIDs := column.NewColumnVarChar("id", []string{"doc1", "doc2"})
			Mock(mockMilvusClient.Insert).Return(mockIDs, nil).Build()
			mockConf := &IndexerConfig{
				Client:     mockMilvusClient,
				Collection: defaultCollection,
				Embedding:  mockemb,
				Dim:        4,
			}
			indexer, err := NewIndexer(ctx, mockConf)

			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)

			// 测试成功存储的情况
			ids, err := indexer.Store(ctx, docs, WithUpsert())
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.Store] failed to upsert documents: upsert failed"))
			convey.So(ids, convey.ShouldBeNil)

		})
		PatchConvey("test with insert error", func() {
			//模拟Insert失败的情况
			Mock((*milvusclient.Client).Insert).Return(milvusclient.InsertResult{}, fmt.Errorf("insert failed")).Build()
			mockConf := &IndexerConfig{
				Client:     mockMilvusClient,
				Collection: defaultCollection,
				Embedding:  mockemb,
				Dim:        4,
			}
			indexer, err := NewIndexer(ctx, mockConf)

			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)

			// 测试成功存储的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.Store] failed to insert documents: insert failed"))
			convey.So(ids, convey.ShouldBeNil)
		})

	})
}

func TestIndexer_NewIndexer(t *testing.T) {
	PatchConvey("test NewIndexer", t, func() {
		ctx := context.Background()
		mockMilvusClient := &milvusclient.Client{}
		mockemb := &mockEmbedding{
			vectors: [][]float64{
				{0.1, 0.2, 0.3, 0.4},
				{0.5, 0.6, 0.7, 0.8},
			},
			err: nil,
		}
		PatchConvey("test NewIndexer with nil config", func() {
			Mock((*IndexerConfig).validate).To(func(conf *IndexerConfig) error {
				return fmt.Errorf("config is nil")
			}).Build()

			indexer, err := NewIndexer(ctx, nil)
			convey.So(indexer, convey.ShouldBeNil)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "invalid indexer config")
		})

		PatchConvey("test NewIndexer with config validation error TestIndexer_NewIndexer- nil client", func() {
			// 创建一个mock配置
			mockConf := &IndexerConfig{
				Client: nil,
				Dim:    128,
			}

			indexer, err := NewIndexer(ctx, mockConf)
			convey.So(indexer, convey.ShouldBeNil)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "the milvus client is nil")
		})

		PatchConvey("test NewIndexer with config validation error - invalid dim", func() {
			// 创建一个mock milvus client
			mockClient := &milvusclient.Client{}

			mockConf := &IndexerConfig{
				Client: mockClient,
				Dim:    0, // 故意设置为0触发错误
			}

			indexer, err := NewIndexer(ctx, mockConf)
			convey.So(indexer, convey.ShouldBeNil)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "the dimension of the vector must be greater than 0")
		})

		PatchConvey("test NewIndexer with valid config", func() {
			// 创建一个mock milvus client
			mockClient := &milvusclient.Client{}
			mockEmbedding := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:    mockClient,
				Dim:       128,
				Embedding: mockEmbedding,
			}

			// 使用monkey patch模拟ensureCollation的行为
			Mock((*IndexerConfig).ensureCollation).To(func(conf *IndexerConfig, ctx context.Context) error {
				return nil // 模拟成功创建collection
			}).Build()

			indexer, err := NewIndexer(ctx, mockConf)
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			convey.So(indexer.conf, convey.ShouldEqual, mockConf)
		})

		PatchConvey("test NewIndexer with ensureCollation error", func() {
			// 创建一个mock milvus client
			mockClient := &milvusclient.Client{}

			mockConf := &IndexerConfig{
				Client: mockClient,
				Dim:    128,
			}
			Mock((*IndexerConfig).ensureCollation).To(func(conf *IndexerConfig, ctx context.Context) error {
				return fmt.Errorf("failed to ensure collection")
			}).Build()

			indexer, err := NewIndexer(ctx, mockConf)
			convey.So(indexer, convey.ShouldBeNil)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "failed to ensure collection")
		})
		PatchConvey("test create collection error", func() {
			Mock(GetMethod(mockMilvusClient, "HasCollection")).Return(false, nil).Build()
			Mock(GetMethod(mockMilvusClient, "CreateCollection")).Return(fmt.Errorf("create collection failed")).Build()
			mockConf := &IndexerConfig{
				Client:     mockMilvusClient,
				Collection: defaultCollection,
				Embedding:  mockemb,
				Dim:        4,
			}
			indexer, err := NewIndexer(ctx, mockConf)
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.NewIndexer] failed to ensure collection: [Indexer.NewIndexer] failed to create collection: create collection failed"))
			convey.So(indexer, convey.ShouldBeNil)
		})

	})
}
