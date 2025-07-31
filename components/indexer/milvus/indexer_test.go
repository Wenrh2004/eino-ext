/*
 * Copyright 2025 CloudWeGo Authors
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

package milvus

import (
	"context"
	"fmt"
	"testing"
	
	. "github.com/bytedance/mockey"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/smartystreets/goconvey/convey"
)

// 模拟Embedding实现
type mockEmbedding struct{}

func (m *mockEmbedding) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	result := make([][]float64, len(texts))
	for i := range texts {
		result[i] = []float64{0.1, 0.2, 0.3}
	}
	return result, nil
}

func TestNewIndexer(t *testing.T) {
	PatchConvey("test NewIndexer", t, func() {
		ctx := context.Background()
		Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
		mockClient, _ := client.NewClient(ctx, client.Config{})
		mockEmb := &mockEmbedding{}
		
		PatchConvey("test indexer config check", func() {
			PatchConvey("test client not provided", func() {
				i, err := NewIndexer(ctx, &IndexerConfig{
					Client:     nil,
					Collection: "",
					Embedding:  mockEmb,
				})
				convey.So(err, convey.ShouldBeError, fmt.Errorf("[NewIndexer] milvus client not provided"))
				convey.So(i, convey.ShouldBeNil)
			})
			
			PatchConvey("test embedding not provided", func() {
				i, err := NewIndexer(ctx, &IndexerConfig{
					Client:     mockClient,
					Collection: "",
					Embedding:  nil,
				})
				convey.So(err, convey.ShouldBeError, fmt.Errorf("[NewIndexer] embedding not provided"))
				convey.So(i, convey.ShouldBeNil)
			})
			
			PatchConvey("test partition name set and partition num lager than 1", func() {
				i, err := NewIndexer(ctx, &IndexerConfig{
					Client:        mockClient,
					Collection:    "",
					PartitionNum:  2,
					PartitionName: "_default",
					Embedding:     mockEmb,
				})
				convey.So(err, convey.ShouldEqual, fmt.Errorf("[NewIndexer] not support manually specifying the partition names if partition key mode is used"))
				convey.So(i, convey.ShouldBeNil)
			})
		})
		
		PatchConvey("test pre-check", func() {
			Mock(GetMethod(mockClient, "HasCollection")).To(func(ctx context.Context, collName string) (bool, error) {
				if collName != defaultCollection {
					return false, nil
				}
				return true, nil
			}).Build()
			
			PatchConvey("test collection not found", func() {
				// 模拟创建集合
				Mock(GetMethod(mockClient, "CreateCollection")).Return(nil).Build()
				
				// 模拟描述集合
				Mock(GetMethod(mockClient, "DescribeCollection")).To(func(ctx context.Context, collName string) (*entity.Collection, error) {
					if collName != defaultCollection {
						return nil, fmt.Errorf("collection not found")
					}
					return &entity.Collection{
						Schema: &entity.Schema{
							Fields: getDefaultFields(),
						},
						Loaded: true,
					}, nil
				}).Build()
				
				i, err := NewIndexer(ctx, &IndexerConfig{
					Client:     mockClient,
					Collection: "test_collection",
					Embedding:  mockEmb,
				})
				convey.So(err, convey.ShouldBeError)
				convey.So(i, convey.ShouldBeNil)
				
				PatchConvey("test collection schema check", func() {
					// 模拟集合已存在但schema不匹配
					Mock(GetMethod(mockClient, "DescribeCollection")).To(func(ctx context.Context, collName string) (*entity.Collection, error) {
						return &entity.Collection{
							Schema: &entity.Schema{
								Fields: []*entity.Field{
									{
										Name:     "different_field",
										DataType: entity.FieldTypeInt64,
									},
								},
							},
							Loaded: true,
						}, nil
					}).Build()
					
					i, err := NewIndexer(ctx, &IndexerConfig{
						Client:     mockClient,
						Collection: defaultCollection,
						Fields:     getDefaultFields(),
						Embedding:  mockEmb,
					})
					convey.So(err, convey.ShouldBeError, fmt.Errorf("[NewIndexer] collection schema not match"))
					convey.So(i, convey.ShouldBeNil)
				})
				
				PatchConvey("test collection not loaded", func() {
					// 模拟集合未加载
					Mock(GetMethod(mockClient, "DescribeCollection")).To(func(ctx context.Context, collName string) (*entity.Collection, error) {
						return &entity.Collection{
							Schema: &entity.Schema{
								Fields: getDefaultFields(),
							},
							Loaded: false,
						}, nil
					}).Build()
					
					// 模拟获取加载状态
					Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, nil).Build()
					
					// 模拟描述索引
					Mock(GetMethod(mockClient, "DescribeIndex")).Return([]entity.Index{
						entity.NewGenericIndex("vector", entity.AUTOINDEX, nil),
					}, nil).Build()
					
					// 模拟创建索引
					Mock(GetMethod(mockClient, "CreateIndex")).Return(nil).Build()
					
					// 模拟加载集合
					Mock(GetMethod(mockClient, "LoadCollection")).Return(nil).Build()
					
					i, err := NewIndexer(ctx, &IndexerConfig{
						Client:     mockClient,
						Collection: defaultCollection,
						Embedding:  mockEmb,
					})
					convey.So(err, convey.ShouldBeNil)
					convey.So(i, convey.ShouldNotBeNil)
				})
				
				PatchConvey("test create indexer with custom config", func() {
					// 模拟集合已加载
					Mock(GetMethod(mockClient, "DescribeCollection")).To(func(ctx context.Context, collName string) (*entity.Collection, error) {
						return &entity.Collection{
							Schema: &entity.Schema{
								Fields: getDefaultFields(),
							},
							Loaded: true,
						}, nil
					}).Build()
					
					i, err := NewIndexer(ctx, &IndexerConfig{
						Client:              mockClient,
						Collection:          defaultCollection,
						Description:         "custom description",
						PartitionNum:        0,
						Fields:              getDefaultFields(),
						SharedNum:           1,
						ConsistencyLevel:    defaultConsistencyLevel,
						MetricType:          defaultMetricType,
						Embedding:           mockEmb,
						EnableDynamicSchema: true,
					})
					convey.So(err, convey.ShouldBeNil)
					convey.So(i, convey.ShouldNotBeNil)
				})
				
				PatchConvey("test partition pre-check", func() {
					PatchConvey("test check partition is error", func() {
						Mock(GetMethod(mockClient, "HasPartition")).Return(false, fmt.Errorf("collection not found")).Build()
						i, err := NewIndexer(ctx, &IndexerConfig{
							Client:        mockClient,
							PartitionNum:  0,
							PartitionName: "test",
							Embedding:     mockEmb,
						})
						convey.So(err, convey.ShouldNotBeNil)
						convey.So(i, convey.ShouldBeNil)
					})
					
					PatchConvey("test partition not found", func() {
						Mock(GetMethod(mockClient, "HasPartition")).Return(false, nil).Build()
						PatchConvey("test create partition has error", func() {
							Mock(GetMethod(mockClient, "CreatePartition")).Return(fmt.Errorf("create partition failed")).Build()
							i, err := NewIndexer(ctx, &IndexerConfig{
								Client:        mockClient,
								PartitionNum:  0,
								PartitionName: "test",
								Embedding:     mockEmb,
							})
							convey.So(err, convey.ShouldNotBeNil)
							convey.So(i, convey.ShouldBeNil)
						})
						PatchConvey("test partition loaded", func() {
							Mock(GetMethod(mockClient, "CreatePartition")).Return(nil).Build()
							PatchConvey("test create partition has error", func() {
								Mock(GetMethod(mockClient, "LoadPartitions")).Return(fmt.Errorf("load partition failed")).Build()
								i, err := NewIndexer(ctx, &IndexerConfig{
									Client:        mockClient,
									PartitionNum:  0,
									PartitionName: "test",
									Embedding:     mockEmb,
								})
								convey.So(err, convey.ShouldNotBeNil)
								convey.So(i, convey.ShouldBeNil)
							})
							PatchConvey("test partition loaded success", func() {
								Mock(GetMethod(mockClient, "LoadPartitions")).Return(nil).Build()
								i, err := NewIndexer(ctx, &IndexerConfig{
									Client:        mockClient,
									PartitionNum:  0,
									PartitionName: "test",
									Embedding:     mockEmb,
								})
								convey.So(err, convey.ShouldBeNil)
								convey.So(i, convey.ShouldNotBeNil)
							})
						})
					})
				})
			})
		})
	})
}

func TestIndexer_Store(t *testing.T) {
	PatchConvey("test Indexer.Store", t, func() {
		ctx := context.Background()
		Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
		mockClient, _ := client.NewClient(ctx, client.Config{})
		
		// 模拟集合已加载
		Mock(GetMethod(mockClient, "DescribeCollection")).To(func(ctx context.Context, collName string) (*entity.Collection, error) {
			return &entity.Collection{
				Schema: &entity.Schema{
					Fields: getDefaultFields(),
				},
				Loaded: true,
			}, nil
		}).Build()
		
		// 模拟HasCollection
		Mock(GetMethod(mockClient, "HasCollection")).Return(true, nil).Build()
		
		// 创建测试文档
		docs := []*schema.Document{
			{
				ID:       "doc1",
				Content:  "This is a test document",
				MetaData: map[string]interface{}{"key": "value"},
			},
			{
				ID:       "doc2",
				Content:  "This is another test document",
				MetaData: map[string]interface{}{"key2": "value2"},
			},
		}
		
		PatchConvey("test store with document converter error", func() {
			// 创建带有错误的文档转换器的索引器
			mockEmb := &mockEmbedding{}
			indexer, err := NewIndexer(ctx, &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				Embedding:  mockEmb,
				DocumentConverter: func(ctx context.Context, docs []*schema.Document, vectors [][]float64) ([]interface{}, error) {
					return nil, fmt.Errorf("document converter error")
				},
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			
			// 测试文档转换器错误的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.Store] failed to convert documents: document converter error"))
			convey.So(ids, convey.ShouldBeNil)
		})
		
		PatchConvey("test store with insert rows error", func() {
			// 模拟InsertRows错误
			Mock(GetMethod(mockClient, "InsertRows")).Return(nil, fmt.Errorf("insert rows error")).Build()
			
			// 创建索引器
			mockEmb := &mockEmbedding{}
			indexer, err := NewIndexer(ctx, &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				Embedding:  mockEmb,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			
			// 测试插入行错误的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.Store] failed to insert rows: insert rows error"))
			convey.So(ids, convey.ShouldBeNil)
		})
		
		PatchConvey("test store with flush error", func() {
			// 模拟InsertRows成功
			mockIDs := entity.NewColumnVarChar("id", []string{"doc1", "doc2"})
			Mock(GetMethod(mockClient, "InsertRows")).Return(mockIDs, nil).Build()
			
			// 模拟Flush错误
			Mock(GetMethod(mockClient, "Flush")).Return(fmt.Errorf("flush error")).Build()
			
			// 创建索引器
			mockEmb := &mockEmbedding{}
			indexer, err := NewIndexer(ctx, &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				Embedding:  mockEmb,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			
			// 测试刷新错误的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Indexer.Store] failed to flush collection: flush error"))
			convey.So(ids, convey.ShouldBeNil)
		})
		
		PatchConvey("test store success", func() {
			// 模拟InsertRows成功
			mockIDs := entity.NewColumnVarChar("id", []string{"doc1", "doc2"})
			Mock(GetMethod(mockClient, "InsertRows")).Return(mockIDs, nil).Build()
			
			// 模拟Flush成功
			Mock(GetMethod(mockClient, "Flush")).Return(nil).Build()
			
			// 创建索引器
			mockEmb := &mockEmbedding{}
			indexer, err := NewIndexer(ctx, &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				Embedding:  mockEmb,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			
			// 测试成功存储的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeNil)
			convey.So(ids, convey.ShouldNotBeNil)
			convey.So(len(ids), convey.ShouldEqual, 2)
			convey.So(ids[0], convey.ShouldEqual, "doc1")
			convey.So(ids[1], convey.ShouldEqual, "doc2")
		})
		
		PatchConvey("test store with custom embedding", func() {
			// 模拟InsertRows成功
			mockIDs := entity.NewColumnVarChar("id", []string{"doc1", "doc2"})
			Mock(GetMethod(mockClient, "InsertRows")).Return(mockIDs, nil).Build()
			
			// 模拟Flush成功
			Mock(GetMethod(mockClient, "Flush")).Return(nil).Build()
			
			// 创建索引器
			mockEmb := &mockEmbedding{}
			indexer, err := NewIndexer(ctx, &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				Embedding:  mockEmb,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(indexer, convey.ShouldNotBeNil)
			
			// 测试使用自定义embedding的情况
			ids, err := indexer.Store(ctx, docs)
			convey.So(err, convey.ShouldBeNil)
			convey.So(ids, convey.ShouldNotBeNil)
			convey.So(len(ids), convey.ShouldEqual, 2)
		})
	})
}

func TestIndexer_getSchema(t *testing.T) {
	PatchConvey("test getSchema ", t, func() {

		PatchConvey("All parameters provided", func() {
			defaultFields := getDefaultFields()
			mockConf := &IndexerConfig{
				Collection:  defaultCollection,
				Description: "the collection for eino",
				Fields:      defaultFields,
			}
			schema := mockConf.getSchema(mockConf.Collection, mockConf.Description, mockConf.Fields)
			convey.So(schema, convey.ShouldNotBeNil)
			convey.So(schema.CollectionName, convey.ShouldEqual, defaultCollection)
			convey.So(schema.Description, convey.ShouldEqual, "the collection for eino")
			convey.So(schema.Fields, convey.ShouldResemble, defaultFields)
		})
		PatchConvey("Collection not provided", func() {
			defaultFields := getDefaultFields()
			mockConf := &IndexerConfig{
				Description: "the collection for eino",
				Fields:      defaultFields,
			}
			schema := mockConf.getSchema(mockConf.Collection, mockConf.Description, mockConf.Fields)
			convey.So(schema, convey.ShouldNotBeNil)
			convey.So(schema.CollectionName, convey.ShouldEqual, "")
			convey.So(schema.Description, convey.ShouldEqual, "the collection for eino")
			convey.So(schema.Fields, convey.ShouldResemble, defaultFields)
		})
		PatchConvey("Fields not provided", func() {
			mockConf := &IndexerConfig{
				Collection:  "eino_collection",
				Description: "the collection for eino",
			}
			schema := mockConf.getSchema(mockConf.Collection, mockConf.Description, mockConf.Fields)
			convey.So(schema, convey.ShouldNotBeNil)
			convey.So(schema.CollectionName, convey.ShouldEqual, "eino_collection")
			convey.So(schema.Description, convey.ShouldEqual, "the collection for eino")
			convey.So(schema.Fields, convey.ShouldBeNil)
		})
		PatchConvey("Description not provided", func() {
			defaultFields := getDefaultFields()
			mockConf := &IndexerConfig{
				Collection: "eino_collection",
				Fields:     defaultFields,
			}
			schema := mockConf.getSchema(mockConf.Collection, mockConf.Description, mockConf.Fields)
			convey.So(schema, convey.ShouldNotBeNil)
			convey.So(schema.CollectionName, convey.ShouldEqual, "eino_collection")
			convey.So(schema.Description, convey.ShouldEqual, "")
			convey.So(schema.Fields, convey.ShouldResemble, defaultFields)
		})
	})
}

func TestIndexer_getDefaultDocumentConvert(t *testing.T) {
	PatchConvey("test getDefaultDocumentConvert", t, func() {
		ctx := context.Background()

		PatchConvey("test convert documents with valid input", func() {
			docs := []*schema.Document{
				{
					ID:       "doc1",
					Content:  "This is document 1",
					MetaData: map[string]interface{}{"key1": "value1"},
				},
				{
					ID:       "doc2",
					Content:  "This is document 2",
					MetaData: map[string]interface{}{"key2": "value2", "number": 42},
				},
			}

			vectors := [][]float64{
				{0.1, 0.2, 0.3},
				{0.4, 0.5, 0.6},
			}

			mockConf := &IndexerConfig{}
			converter := mockConf.getDefaultDocumentConvert()
			rows, err := converter(ctx, docs, vectors)

			convey.So(err, convey.ShouldBeNil)
			convey.So(rows, convey.ShouldNotBeNil)
			convey.So(len(rows), convey.ShouldEqual, 2)

			row1, ok := rows[0].(*defaultSchema)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(row1.ID, convey.ShouldEqual, "doc1")
			convey.So(row1.Content, convey.ShouldEqual, "This is document 1")
			convey.So(row1.Vector, convey.ShouldNotBeNil)

			row2, ok := rows[1].(*defaultSchema)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(row2.ID, convey.ShouldEqual, "doc2")
			convey.So(row2.Content, convey.ShouldEqual, "This is document 2")
			convey.So(row2.Vector, convey.ShouldNotBeNil)
		})

		PatchConvey("test convert documents with empty input", func() {
			var docs []*schema.Document
			var vectors [][]float64

			mockConf := &IndexerConfig{}
			converter := mockConf.getDefaultDocumentConvert()
			rows, err := converter(ctx, docs, vectors)

			convey.So(err, convey.ShouldBeNil)
			convey.So(rows, convey.ShouldNotBeNil)
			convey.So(len(rows), convey.ShouldEqual, 0)
		})

		PatchConvey("test convert documents with invalid metadata", func() {
			ch := make(chan int)

			docs := []*schema.Document{
				{
					ID:       "doc1",
					Content:  "This is document 1",
					MetaData: map[string]interface{}{"key1": ch},
				},
			}

			vectors := [][]float64{{0.1, 0.2, 0.3}}

			mockConf := &IndexerConfig{}
			converter := mockConf.getDefaultDocumentConvert()
			rows, err := converter(ctx, docs, vectors)

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(rows, convey.ShouldBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "failed to marshal metadata")
		})

		PatchConvey("test convert documents with nil vectors", func() {
			docs := []*schema.Document{
				{
					ID:       "doc1",
					Content:  "This is document 1",
					MetaData: map[string]interface{}{"key1": "value1"},
				},
			}

			var vectors [][]float64 = nil

			mockConf := &IndexerConfig{}
			converter := mockConf.getDefaultDocumentConvert()
			rows, err := converter(ctx, docs, vectors)

			convey.So(err, convey.ShouldBeNil)
			convey.So(rows, convey.ShouldNotBeNil)
			convey.So(len(rows), convey.ShouldEqual, 0)
		})
	})
}

func TestIndexer_createdDefaultIndex(t *testing.T) {
	PatchConvey("test createdDefaultIndex", t, func() {
		ctx := context.Background()

		PatchConvey("test create default index with valid config", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: defaultMetricType,
			}

			// Mock the CreateIndex method to return nil (success)
			Mock(GetMethod(mockClient, "CreateIndex")).Return(nil).Build()

			err := mockConf.createdDefaultIndex(ctx, false)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test create default index with async mode", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: defaultMetricType,
			}
			Mock(GetMethod(mockClient, "CreateIndex")).Return(nil).Build()

			err := mockConf.createdDefaultIndex(ctx, true)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test create default index with invalid metric type", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: "INVALID_METRIC_TYPE",
			}

			err := mockConf.createdDefaultIndex(ctx, false)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "[NewIndexer] failed to create index")
		})

		PatchConvey("test create default index with client error", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: defaultMetricType,
			}
			Mock(GetMethod(mockClient, "CreateIndex")).Return(fmt.Errorf("create index error")).Build()
			err := mockConf.createdDefaultIndex(ctx, false)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "[NewIndexer] failed to create index")
			convey.So(err.Error(), convey.ShouldContainSubstring, "create index error")
		})
	})
}

func TestIndexer_checkCollectionSchema(t *testing.T) {
	PatchConvey("test checkCollectionSchema", t, func() {

		PatchConvey("test schema fields count not match", func() {
			mockConf := &IndexerConfig{}
			schema := &entity.Schema{
				Fields: []*entity.Field{
					{
						Name:     "field1",
						DataType: entity.FieldTypeInt64,
					},
				},
			}
			fields := []*entity.Field{
				{
					Name:     "field1",
					DataType: entity.FieldTypeInt64,
				},
				{
					Name:     "field2",
					DataType: entity.FieldTypeFloat,
				},
			}
			result := mockConf.checkCollectionSchema(schema, fields)
			convey.So(result, convey.ShouldBeFalse)
		})

		PatchConvey("test schema fields count match but name not match", func() {
			mockConf := &IndexerConfig{}
			schema := &entity.Schema{
				Fields: []*entity.Field{
					{
						Name:     "field1",
						DataType: entity.FieldTypeInt64,
					},
					{
						Name:     "field2",
						DataType: entity.FieldTypeFloat,
					},
				},
			}
			fields := []*entity.Field{
				{
					Name:     "field1",
					DataType: entity.FieldTypeInt64,
				},
				{
					Name:     "field3",
					DataType: entity.FieldTypeFloat,
				},
			}
			result := mockConf.checkCollectionSchema(schema, fields)
			convey.So(result, convey.ShouldBeFalse)
		})

		PatchConvey("test schema fields count match but data type not match", func() {
			mockConf := &IndexerConfig{}
			schema := &entity.Schema{
				Fields: []*entity.Field{
					{
						Name:     "field1",
						DataType: entity.FieldTypeInt64,
					},
					{
						Name:     "field2",
						DataType: entity.FieldTypeFloat,
					},
				},
			}
			fields := []*entity.Field{
				{
					Name:     "field1",
					DataType: entity.FieldTypeInt64,
				},
				{
					Name:     "field2",
					DataType: entity.FieldTypeDouble,
				},
			}
			result := mockConf.checkCollectionSchema(schema, fields)
			convey.So(result, convey.ShouldBeFalse)
		})

		PatchConvey("test schema fields match exactly", func() {
			mockConf := &IndexerConfig{}
			schema := &entity.Schema{
				Fields: []*entity.Field{
					{
						Name:     "field1",
						DataType: entity.FieldTypeInt64,
					},
					{
						Name:     "field2",
						DataType: entity.FieldTypeFloat,
					},
				},
			}
			fields := []*entity.Field{
				{
					Name:     "field1",
					DataType: entity.FieldTypeInt64,
				},
				{
					Name:     "field2",
					DataType: entity.FieldTypeFloat,
				},
			}
			result := mockConf.checkCollectionSchema(schema, fields)
			convey.So(result, convey.ShouldBeTrue)
		})

		PatchConvey("test empty schema and fields", func() {
			mockConf := &IndexerConfig{}
			schema := &entity.Schema{
				Fields: []*entity.Field{},
			}
			var fields []*entity.Field
			result := mockConf.checkCollectionSchema(schema, fields)
			convey.So(result, convey.ShouldBeTrue)
		})
	})
}

func TestIndexer_loadCollection(t *testing.T) {
	PatchConvey("test loadCollection", t, func() {
		ctx := context.Background()

		PatchConvey("test load collection when load state is LoadStateNotExist", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateNotExist
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotExist, nil).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "[NewIndexer] collection not exist")
		})

		PatchConvey("test load collection when load state is LoadStateNotLoad and no index exists", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: defaultMetricType,
			}

			// 模拟获取加载状态返回LoadStateNotLoad
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, nil).Build()

			// 模拟描述索引返回空列表
			Mock(GetMethod(mockClient, "DescribeIndex")).Return([]entity.Index{}, nil).Build()

			// 模拟创建索引成功
			Mock(GetMethod(mockClient, "CreateIndex")).Return(nil).Build()

			// 模拟加载集合成功
			Mock(GetMethod(mockClient, "LoadCollection")).Return(nil).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test load collection when load state is LoadStateNotLoad and index exists", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateNotLoad
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, nil).Build()

			// 模拟描述索引返回已有索引
			Mock(GetMethod(mockClient, "DescribeIndex")).Return([]entity.Index{
				entity.NewGenericIndex("vector", entity.AUTOINDEX, nil),
			}, nil).Build()

			// 模拟加载集合成功
			Mock(GetMethod(mockClient, "LoadCollection")).Return(nil).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test load collection when load state is LoadStateNotLoad and create index failed", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
				MetricType: defaultMetricType,
			}

			// 模拟获取加载状态返回LoadStateNotLoad
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, nil).Build()

			// 模拟描述索引返回空列表
			Mock(GetMethod(mockClient, "DescribeIndex")).Return([]entity.Index{}, nil).Build()

			// 模拟创建索引失败
			Mock(GetMethod(mockClient, "CreateIndex")).Return(fmt.Errorf("create index error")).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "[NewIndexer] failed to create index")
		})

		PatchConvey("test load collection when load state is LoadStateNotLoad and load collection failed", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateNotLoad
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, nil).Build()

			// 模拟描述索引返回已有索引
			Mock(GetMethod(mockClient, "DescribeIndex")).Return([]entity.Index{
				entity.NewGenericIndex("vector", entity.AUTOINDEX, nil),
			}, nil).Build()

			// 模拟加载集合失败
			Mock(GetMethod(mockClient, "LoadCollection")).Return(fmt.Errorf("load collection error")).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "load collection error")
		})

		PatchConvey("test load collection when load state is LoadStateLoading and loading success", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateLoading
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateLoading, nil).Build()

			// 模拟获取加载进度，第一次返回50，第二次返回100
			callCount := 0
			Mock(GetMethod(mockClient, "GetLoadingProgress")).To(func(ctx context.Context, collectionName string, partitionNames []string) (int64, error) {
				callCount++
				if callCount == 1 {
					return 50, nil
				}
				return 100, nil
			}).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test load collection when load state is LoadStateLoading and get loading progress failed", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateLoading
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateLoading, nil).Build()

			// 模拟获取加载进度失败
			Mock(GetMethod(mockClient, "GetLoadingProgress")).Return(int64(0), fmt.Errorf("get loading progress error")).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "get loading progress error")
		})

		PatchConvey("test load collection when load state is LoadStateLoaded", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态返回LoadStateLoaded
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateLoaded, nil).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldBeNil)
		})

		PatchConvey("test load collection when get load state failed", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(ctx, client.Config{})

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Collection: defaultCollection,
			}

			// 模拟获取加载状态失败
			Mock(GetMethod(mockClient, "GetLoadState")).Return(entity.LoadStateNotLoad, fmt.Errorf("get load state error")).Build()

			err := mockConf.loadCollection(ctx)
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "[NewIndexer] failed to get load state")
		})
	})
}

func TestIndexer_check(t *testing.T) {
	PatchConvey("test check", t, func() {

		PatchConvey("test client not provided", func() {
			mockConf := &IndexerConfig{
				Client: nil,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "[NewIndexer] milvus client not provided")
		})

		PatchConvey("test embedding not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})

			mockConf := &IndexerConfig{
				Client:    mockClient,
				Embedding: nil,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "[NewIndexer] embedding not provided")
		})

		PatchConvey("test partition num larger than 1 and partition name provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:        mockClient,
				Embedding:     mockEmb,
				PartitionNum:  2,
				PartitionName: "_default",
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldEqual, "[NewIndexer] not support manually specifying the partition names if partition key mode is used")
		})

		PatchConvey("test collection not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Embedding:  mockEmb,
				Collection: "",
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.Collection, convey.ShouldEqual, defaultCollection)
		})

		PatchConvey("test description not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:      mockClient,
				Embedding:   mockEmb,
				Description: "",
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.Description, convey.ShouldEqual, defaultDescription)
		})

		PatchConvey("test shared num less than or equal to 0", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:    mockClient,
				Embedding: mockEmb,
				SharedNum: 0,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.SharedNum, convey.ShouldEqual, 1)
		})

		PatchConvey("test consistency level out of range", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:           mockClient,
				Embedding:        mockEmb,
				ConsistencyLevel: 6, // 超出有效范围1-5
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.ConsistencyLevel, convey.ShouldEqual, defaultConsistencyLevel)
		})

		PatchConvey("test consistency level less than or equal to 0", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:           mockClient,
				Embedding:        mockEmb,
				ConsistencyLevel: 0,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.ConsistencyLevel, convey.ShouldEqual, defaultConsistencyLevel)
		})

		PatchConvey("test metric type not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:     mockClient,
				Embedding:  mockEmb,
				MetricType: "",
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.MetricType, convey.ShouldEqual, defaultMetricType)
		})

		PatchConvey("test partition num less than or equal to 1", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:       mockClient,
				Embedding:    mockEmb,
				PartitionNum: 1,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.PartitionNum, convey.ShouldEqual, 0)
		})

		PatchConvey("test fields not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:    mockClient,
				Embedding: mockEmb,
				Fields:    nil,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.Fields, convey.ShouldResemble, getDefaultFields())
		})

		PatchConvey("test document converter not provided", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:            mockClient,
				Embedding:         mockEmb,
				DocumentConverter: nil,
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
			convey.So(mockConf.DocumentConverter, convey.ShouldNotBeNil)
		})

		PatchConvey("test all valid config", func() {
			Mock(client.NewClient).Return(&client.GrpcClient{}, nil).Build()
			mockClient, _ := client.NewClient(context.Background(), client.Config{})
			mockEmb := &mockEmbedding{}

			mockConf := &IndexerConfig{
				Client:           mockClient,
				Embedding:        mockEmb,
				Collection:       "test_collection",
				Description:      "test description",
				SharedNum:        2,
				ConsistencyLevel: ConsistencyLevel(3),
				MetricType:       L2,
				PartitionNum:     0,
				Fields:           getDefaultFields(),
			}

			err := mockConf.check()
			convey.So(err, convey.ShouldBeNil)
		})
	})
}
