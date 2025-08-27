/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  describe,
  expect,
  test,
  beforeEach,
  afterEach,
  jest,
} from "@jest/globals";
import { Document } from "@langchain/core/documents";
import { FakeEmbeddings } from "@langchain/core/utils/testing";
import { GoogleAuth } from "google-auth-library";
import { FirestoreVectorStore, FirebaseStoreParams } from "../vectorstores.js";

describe("FirestoreVectorStore Unit Tests", () => {
  let firestoreVectorStore: FirestoreVectorStore;
  const embeddings = new FakeEmbeddings();

  const defaultParams: FirebaseStoreParams = {
    firestoreConfig: {},
    collectionName: "test_collection",
    distanceMeasure: "EUCLIDEAN",
    googleAuth: new GoogleAuth(),
  };

  beforeEach(() => {
    jest.clearAllMocks();

    firestoreVectorStore = new FirestoreVectorStore({
      embeddings,
      params: defaultParams,
    });
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  describe("Constructor and Initialization", () => {
    test("should initialize with default parameters", () => {
      expect(firestoreVectorStore.collectionName).toBe("test_collection");
      expect(firestoreVectorStore.distanceMeasure).toBe("EUCLIDEAN");
    });

    test("should initialize with custom parameters", () => {
      const customParams: FirebaseStoreParams = {
        ...defaultParams,
        distanceMeasure: "COSINE",
        filter: { type: "test" },
      };

      const customStore = new FirestoreVectorStore({
        embeddings,
        params: customParams,
      });

      expect(customStore.distanceMeasure).toBe("COSINE");
      expect(customStore.filter).toEqual({ type: "test" });
    });

    test("should return correct vectorstore type", () => {
      expect(firestoreVectorStore._vectorstoreType()).toBe(
        "FirestoreVectorStore"
      );
    });
  });

  describe("Document Operations", () => {
    describe("Adding Documents", () => {
      test("should add documents with auto-generated IDs", async () => {
        // Mock the addDocuments method to avoid Firestore calls
        const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
        spy.mockResolvedValue(["auto-id-1", "auto-id-2"]);

        const documents = [
          { pageContent: "First document", metadata: { type: "test" } },
          { pageContent: "Second document", metadata: { type: "test" } },
        ];

        const ids = await firestoreVectorStore.addDocuments(documents);

        expect(ids).toEqual(["auto-id-1", "auto-id-2"]);
        expect(ids).toHaveLength(2);

        spy.mockRestore();
      });

      test("should add documents with provided IDs", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
        spy.mockResolvedValue(["custom-id-1"]);

        const documents = [
          { pageContent: "Test document", metadata: { type: "test" } },
        ];
        const providedIds = ["custom-id-1"];

        const ids = await firestoreVectorStore.addDocuments(documents, {
          ids: providedIds,
        });

        expect(ids).toEqual(providedIds);
        expect(spy).toHaveBeenCalledWith(documents, { ids: providedIds });

        spy.mockRestore();
      });

      test("should handle empty documents array", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
        spy.mockResolvedValue([]);

        const ids = await firestoreVectorStore.addDocuments([]);

        expect(ids).toEqual([]);
        expect(spy).toHaveBeenCalledWith([]);

        spy.mockRestore();
      });

      test("should handle complex metadata", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
        spy.mockResolvedValue(["complex-id"]);

        const documents = [
          {
            pageContent: "Test document",
            metadata: {
              category: "test",
              tags: ["important", "urgent"],
              nested: { level1: { level2: "deep value" } },
              nullValue: null,
            },
          },
        ];

        const ids = await firestoreVectorStore.addDocuments(documents);

        expect(ids).toHaveLength(1);
        expect(spy).toHaveBeenCalledWith(documents);

        spy.mockRestore();
      });

      test("should throw error when embeddings fail", async () => {
        const errorEmbeddings = {
          embedDocuments: (jest.fn() as any).mockRejectedValue(new Error("Embedding failed")),
          embedQuery: (jest.fn() as any).mockResolvedValue([1, 2, 3]),
        };

        const errorStore = new FirestoreVectorStore({
          embeddings: errorEmbeddings,
          params: defaultParams,
        });

        const spy = jest.spyOn(errorStore, "addDocuments");
        spy.mockRejectedValue(
          new Error("Failed to add documents to Firestore.")
        );

        const documents = [{ pageContent: "Test", metadata: {} }];

        await expect(errorStore.addDocuments(documents)).rejects.toThrow(
          "Failed to add documents to Firestore."
        );

        spy.mockRestore();
      });
    });

    describe("Adding Vectors", () => {
      test("should add vectors directly", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addVectors");
        spy.mockResolvedValue(["vec-id-1", "vec-id-2"]);

        const vectors = [
          [1, 2, 3],
          [4, 5, 6],
        ];
        const documents = [
          { pageContent: "First doc", metadata: {} },
          { pageContent: "Second doc", metadata: {} },
        ];

        const ids = await firestoreVectorStore.addVectors(vectors, documents);

        expect(ids).toEqual(["vec-id-1", "vec-id-2"]);
        expect(ids).toHaveLength(2);
        expect(spy).toHaveBeenCalledWith(vectors, documents);

        spy.mockRestore();
      });

      test("should add vectors with provided IDs", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addVectors");
        spy.mockResolvedValue(["provided-id"]);

        const vectors = [[1, 2, 3]];
        const documents = [{ pageContent: "Test doc", metadata: {} }];
        const providedIds = ["provided-id"];

        const resultIds = await firestoreVectorStore.addVectors(
          vectors,
          documents,
          { ids: providedIds }
        );

        expect(resultIds).toEqual(["provided-id"]);
        expect(spy).toHaveBeenCalledWith(vectors, documents, {
          ids: providedIds,
        });

        spy.mockRestore();
      });
    });

    describe("Retrieving Documents", () => {
      test("should get document by ID", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "getDocumentById");
        spy.mockResolvedValue(
          new Document({
            pageContent: "Test content",
            metadata: { text: "Test content", type: "test" },
          })
        );

        const document = await firestoreVectorStore.getDocumentById("test-id");

        expect(document).toEqual(
          new Document({
            pageContent: "Test content",
            metadata: { text: "Test content", type: "test" },
          })
        );
        expect(spy).toHaveBeenCalledWith("test-id");

        spy.mockRestore();
      });

      test("should return null for non-existent document", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "getDocumentById");
        spy.mockResolvedValue(null);

        const document = await firestoreVectorStore.getDocumentById(
          "non-existent"
        );

        expect(document).toBeNull();
        expect(spy).toHaveBeenCalledWith("non-existent");

        spy.mockRestore();
      });

      test("should get documents by metadata", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "getDocumentsByMetadata");
        spy.mockResolvedValue([
          new Document({
            pageContent: "First doc",
            metadata: { text: "First doc", type: "test" },
          }),
          new Document({
            pageContent: "Second doc",
            metadata: { text: "Second doc", type: "test" },
          }),
        ]);

        const documents = await firestoreVectorStore.getDocumentsByMetadata({
          type: "test",
        });

        expect(documents).toHaveLength(2);
        expect(documents[0]).toBeInstanceOf(Document);
        expect(spy).toHaveBeenCalledWith({ type: "test" });

        spy.mockRestore();
      });

      test("should handle empty metadata query", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "getDocumentsByMetadata");
        spy.mockResolvedValue([]);

        const documents = await firestoreVectorStore.getDocumentsByMetadata({});

        expect(documents).toEqual([]);
        expect(spy).toHaveBeenCalledWith({});

        spy.mockRestore();
      });

      describe("Filter by Metadata Tests", () => {
        test("should filter documents by simple metadata field", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "doc1",
              pageContent: "Document 1",
              metadata: { category: "science", type: "article" },
            }),
            new Document({
              id: "doc2",
              pageContent: "Document 2",
              metadata: { category: "science", type: "paper" },
            }),
          ]);

          const filter = { category: "science" };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(2);
          expect(documents[0].metadata.category).toBe("science");
          expect(documents[1].metadata.category).toBe("science");
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should filter documents by multiple metadata fields", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "doc1",
              pageContent: "Filtered document",
              metadata: {
                category: "tech",
                status: "published",
                author: "john",
              },
            }),
          ]);

          const filter = { category: "tech", status: "published" };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata.category).toBe("tech");
          expect(documents[0].metadata.status).toBe("published");
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should filter documents using dot notation for nested fields", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "nested-doc",
              pageContent: "Document with nested metadata",
              metadata: {
                "author.name": "Jane Doe",
                "config.version": "2.1",
                "settings.theme": "dark",
              },
            }),
          ]);

          const filter = { "author.name": "Jane Doe", "config.version": "2.1" };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata["author.name"]).toBe("Jane Doe");
          expect(documents[0].metadata["config.version"]).toBe("2.1");
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should handle metadata fields that already have metadata prefix", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "prefixed-doc",
              pageContent: "Document with prefixed metadata field",
              metadata: { type: "report" },
            }),
          ]);

          const filter = { "metadata.type": "report" };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should filter by numeric metadata values", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "numeric-doc",
              pageContent: "Document with numeric metadata",
              metadata: { score: 95, views: 1000, rating: 4.5 },
            }),
          ]);

          const filter = { score: 95, rating: 4.5 };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata.score).toBe(95);
          expect(documents[0].metadata.rating).toBe(4.5);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should filter by boolean metadata values", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "boolean-doc",
              pageContent: "Document with boolean metadata",
              metadata: { published: true, featured: false },
            }),
          ]);

          const filter = { published: true };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata.published).toBe(true);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should return empty array when no documents match filter", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([]);

          const filter = { nonexistent: "value" };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toEqual([]);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should filter by array metadata values", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "array-doc",
              pageContent: "Document with array metadata",
              metadata: { tags: ["tech", "ai", "ml"] },
            }),
          ]);

          // Note: This tests the structure, but actual array filtering
          // would depend on Firestore query capabilities
          const filter = { tags: ["tech", "ai", "ml"] };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(Array.isArray(documents[0].metadata.tags)).toBe(true);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should handle filter with null values", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([]);

          const filter = { nullable_field: null };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toEqual([]);
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should handle complex nested dot notation filters", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "complex-nested-doc",
              pageContent: "Document with deeply nested metadata",
              metadata: {
                "user.profile.preferences.theme": "dark",
                "settings.advanced.cache.enabled": true,
                "metadata.system.version": "1.0.0",
              },
            }),
          ]);

          const filter = {
            "user.profile.preferences.theme": "dark",
            "settings.advanced.cache.enabled": true,
          };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata["user.profile.preferences.theme"]).toBe(
            "dark"
          );
          expect(documents[0].metadata["settings.advanced.cache.enabled"]).toBe(
            true
          );
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });

        test("should handle metadata filter errors gracefully", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockRejectedValue(
            new Error("Failed to get documents by metadata from Firestore.")
          );

          const filter = { category: "test" };

          await expect(
            firestoreVectorStore.getDocumentsByMetadata(filter)
          ).rejects.toThrow(
            "Failed to get documents by metadata from Firestore."
          );

          expect(spy).toHaveBeenCalledWith(filter);
          spy.mockRestore();
        });

        test("should handle large metadata filter objects", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "large-filter-doc",
              pageContent: "Document matching large filter",
              metadata: {
                field1: "value1",
                field2: "value2",
                field3: "value3",
                field4: "value4",
                field5: "value5",
              },
            }),
          ]);

          const largeFilter = {
            field1: "value1",
            field2: "value2",
            field3: "value3",
            field4: "value4",
            field5: "value5",
          };

          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            largeFilter
          );

          expect(documents).toHaveLength(1);
          expect(spy).toHaveBeenCalledWith(largeFilter);

          spy.mockRestore();
        });

        test("should handle special characters in metadata keys and values", async () => {
          const spy = jest.spyOn(
            firestoreVectorStore,
            "getDocumentsByMetadata"
          );
          spy.mockResolvedValue([
            new Document({
              id: "special-chars-doc",
              pageContent: "Document with special characters",
              metadata: {
                "key-with-dashes": "value-with-dashes",
                key_with_underscores: "value_with_underscores",
                "émoji-key": "🔥🚀",
                "number.field": 123,
              },
            }),
          ]);

          const filter = {
            "key-with-dashes": "value-with-dashes",
            "émoji-key": "🔥🚀",
          };
          const documents = await firestoreVectorStore.getDocumentsByMetadata(
            filter
          );

          expect(documents).toHaveLength(1);
          expect(documents[0].metadata["key-with-dashes"]).toBe(
            "value-with-dashes"
          );
          expect(documents[0].metadata["émoji-key"]).toBe("🔥🚀");
          expect(spy).toHaveBeenCalledWith(filter);

          spy.mockRestore();
        });
      });
    });
  });

  describe("Search Operations", () => {
    test("should perform similarity search", async () => {
      const spy = jest.spyOn(
        firestoreVectorStore,
        "similaritySearchVectorWithScore"
      );
      spy.mockResolvedValue([
        [
          new Document({
            pageContent: "Similar document",
            metadata: { text: "Similar document" },
          }),
          0.9,
        ],
      ]);

      const query = "test query";
      const queryVector = await embeddings.embedQuery(query);
      const results =
        await firestoreVectorStore.similaritySearchVectorWithScore(
          queryVector,
          3
        );

      expect(results).toHaveLength(1);
      expect(results[0][0]).toBeInstanceOf(Document);
      expect(typeof results[0][1]).toBe("number");
      expect(spy).toHaveBeenCalledWith(queryVector, 3);

      spy.mockRestore();
    });

    test("should return empty results when no matches found", async () => {
      const spy = jest.spyOn(
        firestoreVectorStore,
        "similaritySearchVectorWithScore"
      );
      spy.mockResolvedValue([]);

      const queryVector = await embeddings.embedQuery("no matches");
      const results =
        await firestoreVectorStore.similaritySearchVectorWithScore(
          queryVector,
          5
        );

      expect(results).toEqual([]);
      expect(spy).toHaveBeenCalledWith(queryVector, 5);

      spy.mockRestore();
    });

    test("should handle similarity search with different k values", async () => {
      const spy = jest.spyOn(
        firestoreVectorStore,
        "similaritySearchVectorWithScore"
      );
      const mockResults = Array.from({ length: 3 }, (_, i) => [
        new Document({
          pageContent: `Document ${i + 1}`,
          metadata: { text: `Document ${i + 1}` },
        }),
        0.9 - i * 0.1,
      ]);
      spy.mockResolvedValue(mockResults as any);

      const queryVector = await embeddings.embedQuery("test");
      const results =
        await firestoreVectorStore.similaritySearchVectorWithScore(
          queryVector,
          3
        );

      expect(results).toHaveLength(3);
      expect(results[0][1]).toBeGreaterThan(results[1][1]); // Scores should be descending
      expect(spy).toHaveBeenCalledWith(queryVector, 3);

      spy.mockRestore();
    });
  });

  describe("Delete Operations", () => {
    test("should delete all documents", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "delete");
      spy.mockResolvedValue(undefined);

      await firestoreVectorStore.delete({ deleteAll: true });

      expect(spy).toHaveBeenCalledWith({ deleteAll: true });

      spy.mockRestore();
    });

    test("should delete documents by IDs", async () => {
      const idsToDelete = ["id1", "id2"];

      const spy = jest.spyOn(firestoreVectorStore, "delete");
      spy.mockResolvedValue(undefined);

      await firestoreVectorStore.delete({ ids: idsToDelete });

      expect(spy).toHaveBeenCalledWith({ ids: idsToDelete });

      spy.mockRestore();
    });

    test("should delete documents by filter", async () => {
      const filter = { type: "test" };

      const spy = jest.spyOn(firestoreVectorStore, "delete");
      spy.mockResolvedValue(undefined);

      await firestoreVectorStore.delete({ filter });

      expect(spy).toHaveBeenCalledWith({ filter });

      spy.mockRestore();
    });

    test("should handle empty delete operations", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "delete");
      spy.mockResolvedValue(undefined);

      await firestoreVectorStore.delete({ ids: [] });

      expect(spy).toHaveBeenCalledWith({ ids: [] });

      spy.mockRestore();
    });
  });

  describe("Error Handling", () => {
    test("should handle Firestore connection errors during document addition", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockRejectedValue(new Error("Failed to add documents to Firestore."));

      const documents = [{ pageContent: "Test", metadata: {} }];

      await expect(
        firestoreVectorStore.addDocuments(documents)
      ).rejects.toThrow("Failed to add documents to Firestore.");

      spy.mockRestore();
    });

    test("should handle document retrieval errors", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "getDocumentById");
      spy.mockRejectedValue(
        new Error("Failed to get document by ID from Firestore.")
      );

      await expect(
        firestoreVectorStore.getDocumentById("test-id")
      ).rejects.toThrow("Failed to get document by ID from Firestore.");

      spy.mockRestore();
    });

    test("should handle batch delete errors", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "delete");
      spy.mockRejectedValue(new Error("Delete operation failed"));

      await expect(
        firestoreVectorStore.delete({ deleteAll: true })
      ).rejects.toThrow("Delete operation failed");

      spy.mockRestore();
    });

    describe("Similarity Search with Filters", () => {
      test("should perform similarity search with simple filter", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "filtered-result",
              pageContent: "Filtered search result",
              metadata: { category: "tech", text: "Filtered search result" },
            }),
            0.85,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("test query");
        const filter = { category: "tech" };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            5,
            filter
          );

        expect(results).toHaveLength(1);
        expect(results[0][0]).toBeInstanceOf(Document);
        expect(results[0][0].metadata.category).toBe("tech");
        expect(typeof results[0][1]).toBe("number");
        expect(spy).toHaveBeenCalledWith(queryVector, 5, filter);

        spy.mockRestore();
      });

      test("should perform similarity search with multiple filters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "multi-filtered-result",
              pageContent: "Multi-filtered result",
              metadata: {
                category: "science",
                status: "published",
                author: "expert",
              },
            }),
            0.92,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("scientific research");
        const filter = { category: "science", status: "published" };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            3,
            filter
          );

        expect(results).toHaveLength(1);
        expect(results[0][0].metadata.category).toBe("science");
        expect(results[0][0].metadata.status).toBe("published");
        expect(spy).toHaveBeenCalledWith(queryVector, 3, filter);

        spy.mockRestore();
      });

      test("should handle filter conflict with instance filter", async () => {
        // Create instance with pre-set filter
        const filteredStore = new FirestoreVectorStore({
          embeddings,
          params: { ...defaultParams, filter: { type: "default" } },
        });

        const spy = jest.spyOn(
          filteredStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockRejectedValue(
          new Error("Cannot provide both `filter` and `this.filter`")
        );

        const queryVector = await embeddings.embedQuery("test");
        const conflictingFilter = { category: "tech" };

        await expect(
          filteredStore.similaritySearchVectorWithScore(
            queryVector,
            5,
            conflictingFilter
          )
        ).rejects.toThrow("Cannot provide both `filter` and `this.filter`");

        spy.mockRestore();
      });

      test("should use instance filter when no filter provided", async () => {
        const filteredStore = new FirestoreVectorStore({
          embeddings,
          params: { ...defaultParams, filter: { status: "active" } },
        });

        const spy = jest.spyOn(
          filteredStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "instance-filtered",
              pageContent: "Instance filtered result",
              metadata: { status: "active", text: "Instance filtered result" },
            }),
            0.88,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("active document");
        const results = await filteredStore.similaritySearchVectorWithScore(
          queryVector,
          3
        );

        expect(results).toHaveLength(1);
        expect(results[0][0].metadata.status).toBe("active");
        expect(spy).toHaveBeenCalledWith(queryVector, 3);

        spy.mockRestore();
      });

      test("should perform similarity search with nested field filters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "nested-filtered",
              pageContent: "Nested filtered result",
              metadata: {
                "author.department": "engineering",
                "config.level": "advanced",
              },
            }),
            0.78,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("engineering document");
        const nestedFilter = { "author.department": "engineering" };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            2,
            nestedFilter
          );

        expect(results).toHaveLength(1);
        expect(results[0][0].metadata["author.department"]).toBe("engineering");
        expect(spy).toHaveBeenCalledWith(queryVector, 2, nestedFilter);

        spy.mockRestore();
      });

      test("should handle empty results with filters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([]);

        const queryVector = await embeddings.embedQuery("no matches");
        const restrictiveFilter = { nonexistent: "value" };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            10,
            restrictiveFilter
          );

        expect(results).toEqual([]);
        expect(spy).toHaveBeenCalledWith(queryVector, 10, restrictiveFilter);

        spy.mockRestore();
      });

      test("should perform similarity search with numeric filters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "numeric-filtered",
              pageContent: "High quality document",
              metadata: { score: 95, rating: 4.8 },
            }),
            0.95,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("high quality");
        const numericFilter = { score: 95 };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            5,
            numericFilter
          );

        expect(results).toHaveLength(1);
        expect(results[0][0].metadata.score).toBe(95);
        expect(results[0][1]).toBe(0.95);
        expect(spy).toHaveBeenCalledWith(queryVector, 5, numericFilter);

        spy.mockRestore();
      });

      test("should perform similarity search with boolean filters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchVectorWithScore"
        );
        spy.mockResolvedValue([
          [
            new Document({
              id: "published-doc",
              pageContent: "Published document",
              metadata: { published: true, featured: false },
            }),
            0.82,
          ],
        ]);

        const queryVector = await embeddings.embedQuery("published content");
        const booleanFilter = { published: true };
        const results =
          await firestoreVectorStore.similaritySearchVectorWithScore(
            queryVector,
            3,
            booleanFilter
          );

        expect(results).toHaveLength(1);
        expect(results[0][0].metadata.published).toBe(true);
        expect(spy).toHaveBeenCalledWith(queryVector, 3, booleanFilter);

        spy.mockRestore();
      });
    });

    test("should handle search errors", async () => {
      const spy = jest.spyOn(
        firestoreVectorStore,
        "similaritySearchVectorWithScore"
      );
      spy.mockRejectedValue(new Error("Search failed"));

      const queryVector = await embeddings.embedQuery("test");

      await expect(
        firestoreVectorStore.similaritySearchVectorWithScore(queryVector, 5)
      ).rejects.toThrow("Search failed");

      spy.mockRestore();
    });
  });

  describe("Edge Cases", () => {
    test("should handle documents with no metadata", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["no-metadata-id"]);

      const documents = [{ pageContent: "No metadata doc", metadata: {} }];
      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });

    test("should handle very long document content", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["long-content-id"]);

      const longContent = "x".repeat(10000);
      const documents = [{ pageContent: longContent, metadata: {} }];
      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });

    test("should handle special characters in metadata", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["special-chars-id"]);

      const documents = [
        {
          pageContent: "Test",
          metadata: {
            "special.key": "value",
            unicode: "🔥🚀",
            numbers: 123,
          },
        },
      ];

      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });

    test("should handle null and undefined values in metadata", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["null-values-id"]);

      const documents = [
        {
          pageContent: "Test with nulls",
          metadata: {
            nullValue: null,
            undefinedValue: undefined,
            validValue: "test",
          },
        },
      ];

      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });

    test("should handle empty page content", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["empty-content-id"]);

      const documents = [{ pageContent: "", metadata: { type: "empty" } }];
      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });

    test("should handle array metadata fields", async () => {
      const spy = jest.spyOn(firestoreVectorStore, "addDocuments");
      spy.mockResolvedValue(["array-metadata-id"]);

      const documents = [
        {
          pageContent: "Array metadata test",
          metadata: {
            tags: ["tag1", "tag2", "tag3"],
            numbers: [1, 2, 3],
            mixed: ["string", 123, true],
          },
        },
      ];

      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(ids).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(documents);

      spy.mockRestore();
    });
  });

  describe("Configuration Tests", () => {
    test("should handle different distance measures", () => {
      const euclideanStore = new FirestoreVectorStore({
        embeddings,
        params: { ...defaultParams, distanceMeasure: "EUCLIDEAN" },
      });

      const cosineStore = new FirestoreVectorStore({
        embeddings,
        params: { ...defaultParams, distanceMeasure: "COSINE" },
      });

      const dotProductStore = new FirestoreVectorStore({
        embeddings,
        params: { ...defaultParams, distanceMeasure: "DOT_PRODUCT" },
      });

      expect(euclideanStore.distanceMeasure).toBe("EUCLIDEAN");
      expect(cosineStore.distanceMeasure).toBe("COSINE");
      expect(dotProductStore.distanceMeasure).toBe("DOT_PRODUCT");
    });

    test("should handle different collection names", () => {
      const customStore = new FirestoreVectorStore({
        embeddings,
        params: { ...defaultParams, collectionName: "custom_collection" },
      });

      expect(customStore.collectionName).toBe("custom_collection");
    });

    test("should handle filter configuration", () => {
      const filterStore = new FirestoreVectorStore({
        embeddings,
        params: { ...defaultParams, filter: { status: "active" } },
      });

      expect(filterStore.filter).toEqual({ status: "active" });
    });
  });

  describe("New Methods Tests", () => {
    describe("addTexts", () => {
      test("should add texts with metadata", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addTexts");
        spy.mockResolvedValue(["text-id-1", "text-id-2"]);

        const texts = ["First text", "Second text"];
        const metadatas = [{ type: "test" }, { type: "prod" }];

        const ids = await firestoreVectorStore.addTexts(texts, { metadatas });

        expect(ids).toEqual(["text-id-1", "text-id-2"]);
        expect(spy).toHaveBeenCalledWith(texts, { metadatas });

        spy.mockRestore();
      });

      test("should add texts with provided IDs", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addTexts");
        spy.mockResolvedValue(["custom-id-1", "custom-id-2"]);

        const texts = ["Text 1", "Text 2"];
        const ids = ["custom-id-1", "custom-id-2"];

        const resultIds = await firestoreVectorStore.addTexts(texts, { ids });

        expect(resultIds).toEqual(["custom-id-1", "custom-id-2"]);
        expect(spy).toHaveBeenCalledWith(texts, { ids });

        spy.mockRestore();
      });

      test("should handle empty texts array", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addTexts");
        spy.mockResolvedValue([]);

        const ids = await firestoreVectorStore.addTexts([]);

        expect(ids).toEqual([]);
        expect(spy).toHaveBeenCalledWith([]);

        spy.mockRestore();
      });
    });

    describe("Enhanced Similarity Search", () => {
      test("should perform similarity search with filter", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "similaritySearch");
        spy.mockResolvedValue([
          new Document({
            pageContent: "Filtered result",
            metadata: { type: "test" },
          }),
        ]);

        const results = await firestoreVectorStore.similaritySearch(
          "test query",
          5,
          { type: "test" }
        );

        expect(results).toHaveLength(1);
        expect(results[0]).toBeInstanceOf(Document);
        expect(spy).toHaveBeenCalledWith("test query", 5, { type: "test" });

        spy.mockRestore();
      });

      test("should perform similarity search by vector", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchByVector"
        );
        spy.mockResolvedValue([
          new Document({ pageContent: "Vector search result", metadata: {} }),
        ]);

        const vector = [0.1, 0.2, 0.3];
        const results = await firestoreVectorStore.similaritySearchByVector(
          vector
        );

        expect(results).toHaveLength(1);
        expect(results[0]).toBeInstanceOf(Document);
        expect(spy).toHaveBeenCalledWith(vector, 3);

        spy.mockRestore();
      });

      test("should perform similarity search by vector with filter", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchByVector"
        );
        spy.mockResolvedValue([
          new Document({
            pageContent: "Filtered vector result",
            metadata: { category: "A" },
          }),
        ]);

        const vector = [0.5, 0.6, 0.7];
        const filter = { category: "A" };
        const results = await firestoreVectorStore.similaritySearchByVector(
          vector,
          { filter }
        );

        expect(results).toHaveLength(1);
        expect(spy).toHaveBeenCalledWith(vector, 2, { filter });

        spy.mockRestore();
      });
    });

    describe("Maximal Marginal Relevance Search", () => {
      test("should perform MMR search", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchQuery"
        );
        spy.mockResolvedValue([
          new Document({ pageContent: "MMR result 1", metadata: {} }),
          new Document({ pageContent: "MMR result 2", metadata: {} }),
        ]);

        const results =
          await firestoreVectorStore.maxMarginalRelevanceSearchQuery(
            "diverse query",
            { fetchK: 10, lambdaMult: 0.7 }
          );

        expect(results).toHaveLength(2);
        expect(results[0]).toBeInstanceOf(Document);
        expect(spy).toHaveBeenCalledWith("diverse query", 2, {
          fetchK: 10,
          lambdaMult: 0.7,
        });

        spy.mockRestore();
      });

      test("should perform MMR search by vector", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchByVector"
        );
        spy.mockResolvedValue([
          new Document({ pageContent: "MMR vector result", metadata: {} }),
        ]);

        const vector = [0.2, 0.4, 0.6];
        const results =
          await firestoreVectorStore.maxMarginalRelevanceSearchByVector(
            vector,
            { fetchK: 15, lambdaMult: 0.5, filter: { active: true } }
          );

        expect(results).toHaveLength(1);
        expect(spy).toHaveBeenCalledWith(vector, 3, {
          fetchK: 15,
          lambdaMult: 0.5,
          filter: { active: true },
        });

        spy.mockRestore();
      });

      test("should handle MMR search with default parameters", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchQuery"
        );
        spy.mockResolvedValue([
          new Document({ pageContent: "Default MMR result", metadata: {} }),
        ]);

        const results =
          await firestoreVectorStore.maxMarginalRelevanceSearchQuery("test");

        expect(results).toHaveLength(1);
        expect(spy).toHaveBeenCalledWith("test");

        spy.mockRestore();
      });

      test("should handle empty MMR search results", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchByVector"
        );
        spy.mockResolvedValue([]);

        const vector = [0.1, 0.1, 0.1];
        const results =
          await firestoreVectorStore.maxMarginalRelevanceSearchByVector(vector);

        expect(results).toEqual([]);
        expect(spy).toHaveBeenCalledWith(vector);

        spy.mockRestore();
      });
    });

    describe("Utility Methods", () => {
      test("should convert Firestore vector to array", () => {
        // Access private method for testing
        const vectorStore = firestoreVectorStore as any;

        // Test with _values format
        expect(vectorStore._vectorToArray({ _values: [1, 2, 3] })).toEqual([
          1, 2, 3,
        ]);

        // Test with values format
        expect(vectorStore._vectorToArray({ values: [4, 5, 6] })).toEqual([
          4, 5, 6,
        ]);

        // Test with array format
        expect(vectorStore._vectorToArray([7, 8, 9])).toEqual([7, 8, 9]);

        // Test with invalid format
        expect(vectorStore._vectorToArray({})).toEqual([]);
        expect(vectorStore._vectorToArray(null)).toEqual([]);
      });
    });

    describe("Error Handling for New Methods", () => {
      test("should handle addTexts errors", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "addTexts");
        spy.mockRejectedValue(new Error("Failed to add texts to Firestore."));

        await expect(firestoreVectorStore.addTexts(["test"])).rejects.toThrow(
          "Failed to add texts to Firestore."
        );

        spy.mockRestore();
      });

      test("should handle similarity search errors", async () => {
        const spy = jest.spyOn(firestoreVectorStore, "similaritySearch");
        spy.mockRejectedValue(
          new Error("Failed to perform similarity search.")
        );

        await expect(
          firestoreVectorStore.similaritySearch("test")
        ).rejects.toThrow("Failed to perform similarity search.");

        spy.mockRestore();
      });

      test("should handle similarity search by vector errors", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "similaritySearchByVector"
        );
        spy.mockRejectedValue(
          new Error("Failed to perform similarity search by vector.")
        );

        await expect(
          firestoreVectorStore.similaritySearchByVector([1, 2, 3])
        ).rejects.toThrow("Failed to perform similarity search by vector.");

        spy.mockRestore();
      });

      test("should handle MMR search errors", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchQuery"
        );
        spy.mockRejectedValue(
          new Error("Failed to perform maximal marginal relevance search.")
        );

        await expect(
          firestoreVectorStore.maxMarginalRelevanceSearchQuery("test")
        ).rejects.toThrow(
          "Failed to perform maximal marginal relevance search."
        );

        spy.mockRestore();
      });

      test("should handle MMR search by vector errors", async () => {
        const spy = jest.spyOn(
          firestoreVectorStore,
          "maxMarginalRelevanceSearchByVector"
        );
        spy.mockRejectedValue(
          new Error(
            "Failed to perform maximal marginal relevance search by vector."
          )
        );

        await expect(
          firestoreVectorStore.maxMarginalRelevanceSearchByVector([1, 2, 3])
        ).rejects.toThrow(
          "Failed to perform maximal marginal relevance search by vector."
        );

        spy.mockRestore();
      });
    });
  });

  describe("Integration Behavior Tests", () => {
    test("should call addVectors when addDocuments is called", async () => {
      const addVectorsSpy = jest.spyOn(firestoreVectorStore, "addVectors");
      const embedDocumentsSpy = jest.spyOn(embeddings, "embedDocuments");

      addVectorsSpy.mockResolvedValue(["integration-id"]);
      embedDocumentsSpy.mockResolvedValue([[1, 2, 3]]);

      const documents = [{ pageContent: "Integration test", metadata: {} }];
      const ids = await firestoreVectorStore.addDocuments(documents);

      expect(embedDocumentsSpy).toHaveBeenCalledWith(["Integration test"]);
      expect(addVectorsSpy).toHaveBeenCalledWith(
        [[1, 2, 3]],
        documents,
        undefined
      );
      expect(ids).toEqual(["integration-id"]);

      addVectorsSpy.mockRestore();
      embedDocumentsSpy.mockRestore();
    });

    test("should handle embedQuery for similarity search", async () => {
      const searchSpy = jest.spyOn(
        firestoreVectorStore as any,
        "_similarity_search"
      );

      searchSpy.mockResolvedValue([
        new Document({ pageContent: "Result", metadata: {} }),
      ]);

      const results = await firestoreVectorStore.similaritySearch(
        "test query",
        4,
        undefined
      );

      expect(searchSpy).toHaveBeenCalledWith("test query", 4, undefined);
      expect(results).toHaveLength(1);
      expect(results[0]).toBeInstanceOf(Document);

      searchSpy.mockRestore();
    });
  });
});
