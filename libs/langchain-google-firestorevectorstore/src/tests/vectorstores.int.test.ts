/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, expect, test, beforeAll, afterEach } from "@jest/globals";
import * as uuid from "uuid";
import { GoogleAuth } from "google-auth-library";
import { Document } from "@langchain/core/documents";
import { FakeEmbeddings } from "@langchain/core/utils/testing";
import { FirestoreVectorStore, FirebaseStoreParams } from "../vectorstores.js";

describe("FirestoreVectorStore", () => {
  let firestoreVectorStore: FirestoreVectorStore;
  const collectionName = "testCollection";
  const embeddings = new FakeEmbeddings();
  const googleAuth = new GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
  });

  beforeAll(async () => {
    const firestoreConfig = {};

    const params: FirebaseStoreParams = {
      firestoreConfig,
      collectionName,
      distanceMeasure: "EUCLIDEAN",
      googleAuth,
    };
    firestoreVectorStore = new FirestoreVectorStore({ embeddings, params });

    // clear everything in store before starting any test
    await firestoreVectorStore.delete({
      deleteAll: true,
    });
  });

  afterEach(async () => {
    // Clean up test data after each test for better isolation
    await firestoreVectorStore.delete({
      deleteAll: true,
    });
  });

  describe("Document Operations", () => {
    describe("Adding Documents", () => {
      test("should add documents with user-provided ids", async () => {
        const documentId = uuid.v4();
        const pageContent = "Test content";

        const ids = await firestoreVectorStore.addDocuments(
          [{ pageContent, metadata: {} }],
          {
            ids: [documentId],
          }
        );

        expect(ids).toEqual([documentId]);

        await firestoreVectorStore.addDocuments(
          [{ pageContent: `${pageContent} upserted`, metadata: {} }],
          { ids: [documentId] }
        );

        const updatedDocument = await firestoreVectorStore.getDocumentById(
          documentId
        );
        expect(updatedDocument).toEqual(
          new Document({
            id: documentId,
            metadata: {},
            pageContent: `${pageContent} upserted`,
          })
        );
      });

      test("should add documents with auto-generated ids", async () => {
        const pageContent = "Test content with auto ID";

        const ids = await firestoreVectorStore.addDocuments([
          { pageContent, metadata: { foo: "bar" } },
        ]);

        expect(ids.length).toEqual(1);
        expect(typeof ids[0]).toBe("string");

        const storedDocument = await firestoreVectorStore.getDocumentById(
          ids[0]
        );
        expect(storedDocument).toEqual(
          new Document({
            id: ids[0],
            metadata: { foo: "bar" },
            pageContent,
          })
        );
      });

      test("should handle empty documents array", async () => {
        const result = await firestoreVectorStore.addDocuments([]);
        expect(result).toEqual([]);
      });

      test("should handle document with complex metadata", async () => {
        const pageContent = "Test content with complex metadata";
        const complexMetadata = {
          category: "test",
          tags: ["important", "urgent"],
          nested: {
            level1: {
              level2: "deep value",
            },
          },
          nullValue: null,
          emptyObject: {},
        };

        const ids = await firestoreVectorStore.addDocuments([
          { pageContent, metadata: complexMetadata },
        ]);

        expect(ids.length).toEqual(1);
        const storedDocument = await firestoreVectorStore.getDocumentById(
          ids[0]
        );
        expect(storedDocument?.metadata).toMatchObject({
          category: "test",
          tags: ["important", "urgent"],
          "nested.level1.level2": "deep value",
        });
        expect(storedDocument?.id).toBeDefined();
      });

      test("should generate unique IDs for multiple documents", async () => {
        const documents = [
          { pageContent: "First document", metadata: {} },
          { pageContent: "Second document", metadata: {} },
          { pageContent: "Third document", metadata: {} },
        ];

        const ids = await firestoreVectorStore.addDocuments(documents);

        expect(ids.length).toEqual(3);
        expect(new Set(ids).size).toEqual(3);
        ids.forEach((id) => expect(typeof id).toBe("string"));
      });

      test("should handle batch operations", async () => {
        const batchDocuments = Array.from({ length: 5 }, (_, i) => ({
          pageContent: `Batch document ${i + 1}`,
          metadata: { batchId: "test-batch", index: i },
        }));

        const ids = await firestoreVectorStore.addDocuments(batchDocuments);
        expect(ids.length).toEqual(5);

        const retrievedDocs = await Promise.all(
          ids.map((id) => firestoreVectorStore.getDocumentById(id))
        );

        retrievedDocs.forEach((doc, index) => {
          expect(doc?.pageContent).toEqual(`Batch document ${index + 1}`);
          expect(doc?.metadata.batchId).toEqual("test-batch");
          expect(doc?.metadata.index).toEqual(index);
          expect(doc?.id).toBeDefined();
        });
      });

      test("should preserve original document when upserting with same ID", async () => {
        const documentId = uuid.v4();
        const originalContent = "Original content";
        const updatedContent = "Updated content";

        await firestoreVectorStore.addDocuments(
          [{ pageContent: originalContent, metadata: { version: 1 } }],
          { ids: [documentId] }
        );

        await firestoreVectorStore.addDocuments(
          [{ pageContent: updatedContent, metadata: { version: 2 } }],
          { ids: [documentId] }
        );

        const finalDocument = await firestoreVectorStore.getDocumentById(
          documentId
        );
        expect(finalDocument?.pageContent).toEqual(updatedContent);
        expect(finalDocument?.metadata.version).toEqual(2);
      });

      test("should handle string array metadata correctly", async () => {
        const pageContent = "Test with string array";
        const metadata = {
          categories: ["tech", "ai", "vector"],
          simple: "value",
        };

        const ids = await firestoreVectorStore.addDocuments([
          { pageContent, metadata },
        ]);

        const document = await firestoreVectorStore.getDocumentById(ids[0]);
        expect(document?.metadata.categories).toEqual(["tech", "ai", "vector"]);
        expect(document?.metadata.simple).toEqual("value");
      });

      test("should add vectors directly", async () => {
        const pageContent = "Test content with direct vector addition";
        const vector = [1, 2, 3];
        const documentId = uuid.v4();

        await firestoreVectorStore.addVectors(
          [vector],
          [{ pageContent, metadata: {} }],
          { ids: [documentId] }
        );

        const storedDocument = await firestoreVectorStore.getDocumentById(
          documentId
        );
        expect(storedDocument).toEqual(
          new Document({
            id: documentId,
            metadata: {},
            pageContent,
          })
        );
      });
    });

    describe("Retrieving Documents", () => {
      test("should get document by id", async () => {
        const documentId = uuid.v4();
        const pageContent = "Test content";
        await firestoreVectorStore.addDocuments(
          [{ pageContent, metadata: {} }],
          {
            ids: [documentId],
          }
        );

        const document = await firestoreVectorStore.getDocumentById(documentId);
        expect(document).toEqual(
          new Document({
            id: documentId,
            metadata: {},
            pageContent,
          })
        );
      });

      test("should get documents by metadata", async () => {
        const pageContent = "Test content for metadata query";
        const metadata = { queryTest: "unique-metadata-test" };
        await firestoreVectorStore.addDocuments([{ pageContent, metadata }]);

        const documents = await firestoreVectorStore.getDocumentsByMetadata(
          metadata
        );
        expect(documents.length).toBeGreaterThan(0);
        expect(documents[0].metadata).toMatchObject({
          queryTest: "unique-metadata-test",
        });
        expect(documents[0].id).toBeDefined();
        expect(documents[0].pageContent).toBe(pageContent);
      });

      test("should validate document structure after retrieval", async () => {
        const pageContent = "Validation test content";
        const metadata = { validated: true, score: 95 };

        const ids = await firestoreVectorStore.addDocuments([
          { pageContent, metadata },
        ]);

        const document = await firestoreVectorStore.getDocumentById(ids[0]);
        expect(document).toBeInstanceOf(Document);
        expect(document?.pageContent).toBe(pageContent);
        expect(typeof document?.metadata).toBe("object");
        expect(document?.metadata.validated).toBe(true);
        expect(document?.metadata.score).toBe(95);
        expect(document?.pageContent).toBe(pageContent);
      });

      test("should handle non-existent document ID gracefully", async () => {
        const nonExistentId = "non-existent-id-12345";
        const document = await firestoreVectorStore.getDocumentById(
          nonExistentId
        );
        expect(document).toBeNull();
      });

      test("should handle empty metadata filter", async () => {
        const documents = await firestoreVectorStore.getDocumentsByMetadata({});
        expect(Array.isArray(documents)).toBe(true);
      });
    });
  });

  describe("Search Operations", () => {
    test("should perform similarity search", async () => {
      // Add test document first
      await firestoreVectorStore.addDocuments([
        { pageContent: "This is a test document", metadata: {} },
      ]);

      const queryVector = await embeddings.embedQuery("This is a second.");
      const searchResults =
        await firestoreVectorStore.similaritySearchVectorWithScore(
          queryVector,
          3
        );

      expect(searchResults.length).toBeLessThanOrEqual(3);
    });
  });

  describe("Delete Operations", () => {
    test("should delete all documents", async () => {
      const pageContent = "Test content to delete all";
      const id = uuid.v4();

      await firestoreVectorStore.addDocuments([
        { pageContent, metadata: { foo: id } },
        { pageContent, metadata: { foo: id } },
      ]);

      await firestoreVectorStore.delete({
        deleteAll: true,
      });

      const remainingDocuments =
        await firestoreVectorStore.getDocumentsByMetadata({
          foo: id,
        });

      expect(remainingDocuments.length).toEqual(0);
    });

    test("should delete documents by IDs", async () => {
      const pageContent = "Test content to delete by ID";
      const documentId = uuid.v4();

      await firestoreVectorStore.addDocuments([{ pageContent, metadata: {} }], {
        ids: [documentId],
      });

      await firestoreVectorStore.delete({
        ids: [documentId],
      });

      const deletedDocument = await firestoreVectorStore.getDocumentById(
        documentId
      );
      expect(deletedDocument).toBeNull();
    });

    test("should delete documents by filter", async () => {
      const pageContent = "Test content to delete by filter";
      const filter = { foo: "bar" };

      await firestoreVectorStore.addDocuments([
        { pageContent, metadata: filter },
      ]);

      await firestoreVectorStore.delete({
        filter,
      });

      const remainingDocuments =
        await firestoreVectorStore.getDocumentsByMetadata(filter);
      expect(remainingDocuments.length).toEqual(0);
    });

    test("should handle delete by multiple IDs", async () => {
      const documents = [
        { pageContent: "Doc to delete 1", metadata: {} },
        { pageContent: "Doc to delete 2", metadata: {} },
        { pageContent: "Doc to keep", metadata: {} },
      ];

      const ids = await firestoreVectorStore.addDocuments(documents);
      const idsToDelete = ids.slice(0, 2);
      const idToKeep = ids[2];

      await firestoreVectorStore.delete({ ids: idsToDelete });

      const deletedDoc1 = await firestoreVectorStore.getDocumentById(
        idsToDelete[0]
      );
      const deletedDoc2 = await firestoreVectorStore.getDocumentById(
        idsToDelete[1]
      );
      const keptDoc = await firestoreVectorStore.getDocumentById(idToKeep);

      expect(deletedDoc1).toBeNull();
      expect(deletedDoc2).toBeNull();
      expect(keptDoc).not.toBeNull();
    });
  });
});
