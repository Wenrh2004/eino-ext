package milvus

import (
	"context"
	"fmt"
	"log"
	"testing"

	. "github.com/bytedance/mockey"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/smartystreets/goconvey/convey"
)

func TestRetriever_Retriever(t *testing.T) {
	PatchConvey("test retriever.Retrieve", t, func() {
		ctx := context.Background()
		// mockEmb := &mockEmbedding{}
		mockMilvusClient := &milvusclient.Client{}
		PatchConvey("test embedding error", func() {
			r, _ := NewRetriever(&RetrieverConfig{
				Client:            mockMilvusClient,
				Collection:        "",
				DocumentConverter: nil,
				VectorConverter:   nil,
				TopK:              0,
				Embedding:         &mockEmbedding{err: fmt.Errorf("embedding error")},
			})
			docs, err := r.Retrieve(ctx, "test")
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] embed vectors has error: embedding error"))
			convey.So(docs, convey.ShouldBeNil)
		})
		PatchConvey("test embedding nil", func() {
			r, _ := NewRetriever(&RetrieverConfig{
				Client:            mockMilvusClient,
				Collection:        "",
				DocumentConverter: nil,
				VectorConverter:   nil,
				TopK:              0,
				Embedding:         nil,
			})
			docs, err := r.Retrieve(ctx, "test")
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] embedding is nil"))
			convey.So(docs, convey.ShouldBeNil)
		})
		PatchConvey("test no vectors generated", func() {
			r, _ := NewRetriever(&RetrieverConfig{
				Client:            mockMilvusClient,
				Collection:        "",
				DocumentConverter: nil,
				VectorConverter:   nil,
				TopK:              0,
				Embedding:         &mockEmbedding{sizeForCall: []int{0}, dims: 1},
			})
			docs, err := r.Retrieve(ctx, "test")
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] no vectors generated for the query"))
			convey.So(docs, convey.ShouldBeNil)
		})
		PatchConvey("test vector has error", func() {
			r, _ := NewRetriever(&RetrieverConfig{
				Client:            mockMilvusClient,
				Collection:        "",
				DocumentConverter: nil,
				VectorConverter:   func(vectors [][]float64) ([]entity.Vector, error) { return nil, fmt.Errorf("vector converter error") },
				TopK:              0,
				Embedding:         &mockEmbedding{sizeForCall: []int{1}, dims: 1},
			})
			docs, err := r.Retrieve(ctx, "test")
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] vector has error: vector converter error"))
			convey.So(docs, convey.ShouldBeNil)
		})
		PatchConvey("test embedding success", func() {

			PatchConvey("test search error", func() {
				Mock((*milvusclient.Client).Search).Return(nil, fmt.Errorf("search error")).Build()
				r, _ := NewRetriever(&RetrieverConfig{
					Client:            mockMilvusClient,
					Collection:        "test_collection", // This will cause collection not found error
					DocumentConverter: nil,
					VectorConverter:   nil,
					TopK:              1,
					Embedding:         &mockEmbedding{sizeForCall: []int{1}},
				})
				documents, err := r.Retrieve(ctx, "test")

				convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] query has error: search error"))
				convey.So(documents, convey.ShouldBeNil)
			})
			PatchConvey("test DocumentConverter error", func() {
				// Mock HybridSearch to return successful result
				mockResultSet := &milvusclient.ResultSet{}
				Mock((*milvusclient.Client).HybridSearch).Return([]milvusclient.ResultSet{*mockResultSet}, nil).Build()

				hybridSearch := NewHybridSearchOption("vector_field", 10).
					WithANNSField("vector_field").
					WithFilter("id > 0")
				r, _ := NewRetriever(&RetrieverConfig{
					Client:     mockMilvusClient,
					Collection: "test_collection",
					DocumentConverter: func(result []milvusclient.ResultSet) ([]*schema.Document, error) {
						return nil, fmt.Errorf("document converter error")
					},
					VectorConverter: nil,
					TopK:            1,
					Embedding:       &mockEmbedding{sizeForCall: []int{1}},
				})
				documents, err := r.Retrieve(ctx, "test", WithHybridSearchOption(hybridSearch))

				convey.So(err, convey.ShouldBeError, fmt.Errorf("[Retriever.Retrieve] query has error: document converter error"))
				convey.So(documents, convey.ShouldBeNil)
			})
			PatchConvey("test HybridSearch options", func() {
				hs := NewHybridSearchOption("vector_field", 10)

				hs = hs.WithANNSField("new_vector_field")
				convey.So(hs.annField, convey.ShouldEqual, "new_vector_field")

				hs = hs.WithGroupByField("group_field")
				convey.So(hs.groupByField, convey.ShouldEqual, "group_field")

				hs = hs.WithGroupSize(5)
				convey.So(hs.groupSize, convey.ShouldEqual, 5)

				hs = hs.WithStrictGroupSize(true)
				convey.So(hs.strictGroupSize, convey.ShouldBeTrue)

				hs = hs.WithSearchParam("key", "value")
				convey.So(hs.searchParam["key"], convey.ShouldEqual, "value")

				hs = hs.WithTemplateParam("template_key", "template_value")
				convey.So(hs.templateParams["template_key"], convey.ShouldEqual, "template_value")

				hs = hs.WithOffset(10)
				convey.So(hs.offset, convey.ShouldEqual, 10)

				hs = hs.WithIgnoreGrowing(true)
				convey.So(hs.ignoreGrowing, convey.ShouldBeTrue)
			})

			PatchConvey("test WithLimit option", func() {
				opt := WithLimit(20)
				convey.So(opt, convey.ShouldNotBeNil)
			})

		})

	})

}

func TestRetriever_NewRetriever(t *testing.T) {
	PatchConvey("test NewRetriever", t, func() {
		mockMilvusClient := &milvusclient.Client{}

		PatchConvey("test client is nil", func() {
			r, err := NewRetriever(&RetrieverConfig{
				Client:            nil,
				Collection:        "",
				DocumentConverter: nil,
				VectorConverter:   nil,
				TopK:              0,
				Embedding:         nil,
			})
			convey.So(err, convey.ShouldBeError, fmt.Errorf("[NewRetriever] config validation failed: [Retriever.RetrieverConfig] milvus client is nil"))
			convey.So(r, convey.ShouldBeNil)
		})

		PatchConvey("test default values", func() {
			r, err := NewRetriever(&RetrieverConfig{
				Client: mockMilvusClient,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(r, convey.ShouldNotBeNil)
			convey.So(r.conf.Collection, convey.ShouldEqual, defaultCollection)
			convey.So(r.conf.TopK, convey.ShouldEqual, defaultTopK)
			convey.So(r.conf.DocumentConverter, convey.ShouldNotBeNil)
			convey.So(r.conf.VectorConverter, convey.ShouldNotBeNil)
		})

		PatchConvey("test zero topK", func() {
			r, err := NewRetriever(&RetrieverConfig{
				Client: mockMilvusClient,
				TopK:   0,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(r, convey.ShouldNotBeNil)
			convey.So(r.conf.TopK, convey.ShouldEqual, defaultTopK)
		})

		PatchConvey("test negative topK", func() {
			r, err := NewRetriever(&RetrieverConfig{
				Client: mockMilvusClient,
				TopK:   -1,
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(r, convey.ShouldNotBeNil)
			convey.So(r.conf.TopK, convey.ShouldEqual, defaultTopK)
		})

		PatchConvey("test empty collection", func() {
			r, err := NewRetriever(&RetrieverConfig{
				Client:     mockMilvusClient,
				Collection: "",
			})
			convey.So(err, convey.ShouldBeNil)
			convey.So(r, convey.ShouldNotBeNil)
			convey.So(r.conf.Collection, convey.ShouldEqual, defaultCollection)
		})

		PatchConvey("test empty document converter", func() {
			mockMilvusClient := &milvusclient.Client{}
			config := &RetrieverConfig{
				Client:            mockMilvusClient,
				Collection:        "test_collection",
				DocumentConverter: nil, // Explicitly set to nil
				VectorConverter:   nil,
				TopK:              10,
				Embedding:         nil,
			}

			err := config.validate()
			convey.So(err, convey.ShouldBeNil)
			convey.So(config.DocumentConverter, convey.ShouldNotBeNil)

			r, err := NewRetriever(config)
			convey.So(err, convey.ShouldBeNil)
			convey.So(r, convey.ShouldNotBeNil)
			convey.So(r.conf.DocumentConverter, convey.ShouldNotBeNil)

			docs, err := r.conf.DocumentConverter([]milvusclient.ResultSet{})
			convey.So(err, convey.ShouldBeNil)
			convey.So(docs, convey.ShouldBeNil)
		})

	})
}

type mockEmbedding struct {
	err         error
	cnt         int
	sizeForCall []int
	dims        int
}

func (m *mockEmbedding) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	if m.cnt > len(m.sizeForCall) {
		log.Fatal("unexpected")
	}

	if m.err != nil {
		return nil, m.err
	}

	slice := make([]float64, m.dims)
	for i := range slice {
		slice[i] = 1.1
	}

	r := make([][]float64, m.sizeForCall[m.cnt])
	m.cnt++
	for i := range r {
		r[i] = slice
	}

	return r, nil
}
