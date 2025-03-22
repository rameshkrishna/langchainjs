import { SyntheticEmbeddings } from "@langchain/core/utils/testing";
import { FirestoreVectorStore } from "../vectorstores.js";
import { Document } from "@langchain/core/documents";
import { GoogleAuth } from "google-auth-library";
import { Firestore } from "@google-cloud/firestore";
// import * as uuid from "uuid";

// Test configuration
const COLLECTION_NAME = "testCollectionint";
const EMBEDDING_SIZE = 1024;

// Test documents
const TEST_DOCUMENTS = [
  new Document({
    pageContent: "apple",
    metadata: { type: "fruit", color: "red" },
  }),
  new Document({
    pageContent: "banana",
    metadata: { type: "fruit", color: "yellow" },
  }),
  new Document({
    pageContent: "carrot",
    metadata: { type: "vegetable", color: "orange" },
  }),
];

describe("FirestoreVectorStore Integration Tests", () => {
  let store: FirestoreVectorStore;
  let firestore: Firestore;
  let embeddings: SyntheticEmbeddings;

  beforeAll(async () => {
    // Initialize real Firestore client
    const auth = new GoogleAuth();
    firestore = new Firestore({
      auth,
    });

    const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    if (!credentialsPath) {
      throw new Error(
        "Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. Firestore client cannot be initialized."
      );
    }

    // Initialize embeddings

    embeddings = new SyntheticEmbeddings({ vectorSize: EMBEDDING_SIZE });

    // Create vector store instance
    store = new FirestoreVectorStore({
      embeddings,
      params: {
        collectionName: COLLECTION_NAME,
      },
    });
  });

  afterEach(async () => {
    // Cleanup after each test
    await store.delete({ deleteAll: true });
  });

  afterAll(async () => {
    // Delete test collection
    const collectionRef = firestore.collection(COLLECTION_NAME);
    const documents = await collectionRef.listDocuments();
    await Promise.all(documents.map((doc) => doc.delete()));
  });

  test("should initialize with correct configuration", () => {
    expect(store).toBeInstanceOf(FirestoreVectorStore);
    expect(store.collectionName).toBe(COLLECTION_NAME);
    expect(store.distanceMeasure).toBe("EUCLIDEAN");
  });

  test("should add and retrieve documents", async () => {
    const ids = await store.addDocuments(TEST_DOCUMENTS);

    // console.log(ids);

    expect(ids).toHaveLength(TEST_DOCUMENTS.length);

    // Verify documents in Firestore
    const docs = await store.firestore.collection(COLLECTION_NAME).get();

    expect(docs.size).toBe(TEST_DOCUMENTS.length);

    // Verify document contents
    const retrievedDocs = await Promise.all(
      ids.map((id) => store.getDocumentById(id))
    );

    expect(retrievedDocs.every((doc) => doc !== null)).toBe(true);
  });

  // test("should perform similarity search", async () => {
  //   await store.addDocuments(TEST_DOCUMENTS);
  //   const queryVector = await embeddings.embedQuery("fruit");

  //   const results = await store.similaritySearchVectorWithScore(queryVector, 2);

  //   expect(results).toHaveLength(2);
  //   expect(results[0][0].pageContent).toMatch(/apple|banana/);
  // });

  test("should delete documents by IDs", async () => {
    const ids = await store.addDocuments(TEST_DOCUMENTS);
    const idToDelete = ids[0];

    await store.delete({ ids: [idToDelete] });

    const remainingDocs = await store.firestore
      .collection(COLLECTION_NAME)
      .get();
    const remainingIds = remainingDocs.docs.map((doc) => doc.id);

    expect(remainingIds).not.toContain(idToDelete);
  });

  test("should delete all documents", async () => {
    await store.addDocuments(TEST_DOCUMENTS);
    await store.delete({ deleteAll: true });

    const docs = await store.firestore.collection(COLLECTION_NAME).get();
    expect(docs.size).toBe(0);
  });

  test("should filter documents by metadata", async () => {
    await store.addDocuments(TEST_DOCUMENTS);

    const results = await store.getDocumentsByMetadata({
      "metadata.type": "fruit",
    });

    // console.log(results);

    expect(results).toHaveLength(2);
    expect(results.map((doc) => doc.pageContent)).toEqual(
      expect.arrayContaining(["apple", "banana"])
    );
  });

  test("should respect custom textKey during add and retrieval should be pageContent", async () => {
    const customTextKey = "customContent";

    // Create a new store instance with a custom textKey
    const storeWithTextKey = new FirestoreVectorStore({
      embeddings,
      params: {
        collectionName: COLLECTION_NAME,
        textKey: customTextKey,
      },
    });

    const customDocs = [
      new Document({
        pageContent: "mango",
        metadata: { type: "fruit", color: "yellow" },
      }),
    ];

    const ids = await storeWithTextKey.addDocuments(customDocs);
    const retrieved = await storeWithTextKey.getDocumentById(ids[0]);

    expect(retrieved).not.toBeNull();
    expect(retrieved?.pageContent).toBe("mango");

    // Also verify Firestore stored it under the custom key
    const rawDoc = await storeWithTextKey.firestore
      .collection(COLLECTION_NAME)
      .doc(ids[0])
      .get();

    const rawData = rawDoc.data();
    expect(rawData).toHaveProperty(customTextKey, "mango");
  });

  test("should create instance from documents", async () => {
    const instance = await FirestoreVectorStore.fromDocuments(
      TEST_DOCUMENTS,
      embeddings,
      {
        collectionName: COLLECTION_NAME,
        firestoreConfig: {
          auth: new GoogleAuth(),
        },
      }
    );

    const docs = await instance.firestore.collection(COLLECTION_NAME).get();
    expect(docs.size).toBe(TEST_DOCUMENTS.length);
  });

  test("should handle batch deletions", async () => {
    // Create large batch of documents
    const largeBatch = Array.from(
      { length: 600 },
      (_, i) =>
        new Document({ pageContent: `doc-${i}`, metadata: { batch: "large" } })
    );

    await store.addDocuments(largeBatch);
    await store.delete({ filter: { batch: "large" } });

    const docs = await store.firestore.collection(COLLECTION_NAME).get();
    expect(docs.size).toBe(0);
  });

  // test("should throw error when using both filters", async () => {
  //   await store.addDocuments(TEST_DOCUMENTS);

  //   await expect(
  //     store.similaritySearchVectorWithScore([0.1, 0.2], 2, { color: "red" })
  //   ).rejects.toThrow("Cannot provide both `filter` and `this.filter`");
  // });
});
