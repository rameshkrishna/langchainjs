// eslint-disable-next-line import/no-extraneous-dependencies
import {
  Firestore,
  FieldValue,
  VectorQuery,
  VectorQuerySnapshot,
  CollectionReference,
  Query,
} from "@google-cloud/firestore";
// eslint-disable-next-line import/no-extraneous-dependencies
import * as uuid from "uuid";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { VectorStore } from "@langchain/core/vectorstores";
import {
  AsyncCaller,
  AsyncCallerParams,
} from "@langchain/core/utils/async_caller";
import type { Callbacks } from "@langchain/core/callbacks/manager";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";
import { flatten } from "flat";
import { GoogleAuth } from "google-auth-library";

export interface FirebaseStoreParams extends AsyncCallerParams {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  firestoreConfig?: FirebaseFirestore.Settings;
  collectionName: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filter?: Record<string, any>;
  distanceMeasure?: "EUCLIDEAN" | "COSINE" | "DOT_PRODUCT";
  googleAuth: GoogleAuth;
}

interface AddDocumentsOptions {
  ids?: string[];
}

interface AddTextsOptions {
  ids?: string[];
  metadatas?: object[] | object;
}

interface SearchOptions {
  k?: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filter?: Record<string, any>;
}

interface MaxMarginalRelevanceSearchOptions extends SearchOptions {
  fetchK?: number;
  lambdaMult?: number;
}

interface DeleteParams {
  deleteAll?: boolean;
  ids?: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filter?: Record<string, any>;
}

interface FirestoreDocumentData {
  id: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
  pageContent: string;
  embedding_field: FieldValue;
}

class FirestoreVectorStore extends VectorStore {
  firestore: Firestore;

  collectionName: string;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filter?: Record<string, any>;

  distanceMeasure: "EUCLIDEAN" | "COSINE" | "DOT_PRODUCT";

  caller: AsyncCaller;

  googleAuth: GoogleAuth;

  /**
   * Constructs a new FirestoreVectorStore instance.
   * @param {EmbeddingsInterface} embeddings - The embeddings interface.
   * @param {FirebaseStoreParams} params - The parameters for the Firestore configuration.
   * @param {Object} params.firestoreConfig - Firestore configuration settings.
   * @param {string} params.collectionName - The name of the Firestore collection.
   * @param {Object} [params.filter] - Default filter to apply to queries.
   * @param {string} [params.distanceMeasure="EUCLIDEAN"] - The distance measure for vector similarity search.
   */
  constructor({
    embeddings,
    params,
  }: {
    embeddings: EmbeddingsInterface;
    params: FirebaseStoreParams;
  }) {
    super(embeddings, params);
    const {
      collectionName,
      filter,
      distanceMeasure,
      firestoreConfig,
      googleAuth,
      ...asyncCallerArgs
    } = params;
    this.googleAuth = googleAuth;
    this.collectionName = collectionName;
    this.filter = filter;
    this.distanceMeasure = distanceMeasure ?? "EUCLIDEAN";
    this.caller = new AsyncCaller(asyncCallerArgs);

    this.firestore = new Firestore({ googleAuth, ...firestoreConfig });
  }

  _vectorstoreType(): string {
    return "FirestoreVectorStore";
  }

  /**
   * Helper method for batch processing operations with consistent error handling.
   * @param {T[]} items - Items to process in batches.
   * @param {(batch: FirebaseFirestore.WriteBatch, item: T) => void} processor - Function to process each item.
   * @param {number} batchLimit - Maximum items per batch.
   */
  private async _processBatch<T>(
    items: T[],
    processor: (batch: FirebaseFirestore.WriteBatch, item: T) => void,
    batchLimit: number = 500
  ): Promise<void> {
    let batch = this.firestore.batch();
    let batchCount = 0;

    for (const item of items) {
      processor(batch, item);
      batchCount += 1;

      if (batchCount === batchLimit) {
        await batch.commit();
        batch = this.firestore.batch();
        batchCount = 0;
      }
    }

    // Commit any remaining writes
    if (batchCount > 0) {
      await batch.commit();
    }
  }

  /**
   * Helper method to create Document instances from Firestore data.
   * @param {FirebaseFirestore.DocumentData | undefined} data - Firestore document data.
   * @returns {DocumentInterface} - Created document instance.
   */
  private _createDocumentFromData(
    data: FirebaseFirestore.DocumentData | undefined
  ): DocumentInterface {
    return new Document({
      id: data?.id,
      metadata: data?.metadata || {},
      pageContent: data?.pageContent || "",
    });
  }

  /**
   * Helper method to normalize filter field names.
   * @param {string} key - The filter key.
   * @returns {string} - Normalized field name.
   */
  private _normalizeFilterField(key: string): string {
    return key.startsWith("metadata.") ? key : `metadata.${key}`;
  }

  /**
   * Helper method for consistent error handling.
   * @param {string} operation - The operation that failed.
   * @param {unknown} error - The error that occurred.
   * @throws {Error} - Always throws a standardized error.
   */
  private _handleError(operation: string, error: unknown): never {
    console.error(`Error ${operation}:`, error);
    throw new Error(`Failed to ${operation}.`);
  }

  /**
   * Helper method to validate search parameters.
   * @param {number} k - Number of results to return.
   * @param {Record<string, any>} [filter] - Optional filter.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _validateSearchParams(k: number, filter?: Record<string, any>): void {
    if (k <= 0) {
      throw new Error("Parameter 'k' must be a positive number");
    }
    if (filter && this.filter) {
      throw new Error(
        "Cannot provide both 'filter' parameter and instance filter"
      );
    }
  }

  /**
   * Adds documents to Firestore with their embeddings.
   * @param {Array<Document>} documents - The documents to add.
   * @param {Object} options - Additional options for adding documents.
   * @returns {Promise<Array<string>>} - The IDs of the added documents.
   */
  async addDocuments(
    documents: DocumentInterface[],
    options?: AddDocumentsOptions
  ): Promise<string[]> {
    try {
      if (!documents.length) {
        return [];
      }

      const texts = documents.map(({ pageContent }) => pageContent);
      const vectors = await this.embeddings.embedDocuments(texts);

      return this.addVectors(vectors, documents, options);
    } catch (error) {
      this._handleError("add documents to Firestore", error);
    }
  }

  /**
   * Adds vectors and corresponding documents to Firestore.
   * @param {Array<Array<number>>} vectors - The vectors to add.
   * @param {Array<Document>} documents - The documents to add.
   * @param {Object} options - Additional options for adding vectors.
   * @returns {Promise<Array<string>>} - The IDs of the added documents.
   */
  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[],
    options?: AddDocumentsOptions
  ): Promise<string[]> {
    try {
      if (vectors.length !== documents.length) {
        throw new Error(
          "Vectors and documents arrays must have the same length"
        );
      }

      const ids = options?.ids;
      const documentIds = ids ?? documents.map(() => uuid.v4());

      const batchRequests = vectors.map((values, idx) => {
        const docData = this.prepareDocData(
          documentIds[idx],
          documents[idx],
          values
        );
        return this.setDocData(documentIds[idx], docData);
      });

      await Promise.all(batchRequests);
      return documentIds;
    } catch (error) {
      this._handleError("add vectors to Firestore", error);
    }
  }

  /**
   * Prepares the Firestore document data.
   * @param {string} id - The document ID.
   * @param {DocumentInterface} document - The document.
   * @param {number[]} values - The embedding values.
   * @returns {FirestoreDocumentData} - The prepared document data.
   */
  prepareDocData(
    id: string,
    document: DocumentInterface,
    values: number[]
  ): FirestoreDocumentData {
    const documentMetadata = { ...document.metadata };
    const stringArrays: Record<string, string[]> = {};

    for (const key of Object.keys(documentMetadata)) {
      if (
        Array.isArray(documentMetadata[key]) &&
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        documentMetadata[key].every((el: any) => typeof el === "string")
      ) {
        stringArrays[key] = documentMetadata[key];
        delete documentMetadata[key];
      }
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const flattenedMetadata = flatten(documentMetadata) as Record<string, any>;

    const metadata = {
      ...flattenedMetadata,
      ...stringArrays,
    };

    for (const key of Object.keys(metadata)) {
      if (
        metadata[key] == null ||
        (typeof metadata[key] === "object" &&
          Object.keys(metadata[key]).length === 0)
      ) {
        delete metadata[key];
      }
    }

    return {
      id,
      metadata,
      pageContent: document.pageContent,
      embedding_field: FieldValue.vector(values),
    };
  }

  /**
   * Sets the document data in Firestore.
   * @param {string} id - The document ID.
   * @param {FirestoreDocumentData} docData - The document data.
   * @returns {Promise<void>}
   */
  protected async setDocData(
    id: string,
    docData: FirestoreDocumentData
  ): Promise<void> {
    try {
      const docRef = this.firestore.collection(this.collectionName).doc(id);
      await docRef.set(docData);
    } catch (error) {
      this._handleError("set document data in Firestore", error);
    }
  }

  /**
   * Deletes documents from Firestore based on the given parameters.
   * @param {DeleteParams} params - The parameters for deletion.
   * @throws {Error} - If neither `ids` nor `deleteAll` is provided.
   */
  async delete(params: DeleteParams): Promise<void> {
    const { deleteAll, ids, filter } = params;

    if (deleteAll) {
      await this.deleteAllDocuments();
    } else if (ids) {
      await this.deleteDocumentsByIds(ids);
    } else if (filter) {
      await this.deleteDocumentsByFilter(filter);
    } else {
      throw new Error("Either ids or deleteAll must be provided.");
    }
  }

  /**
   * Deletes all documents in the Firestore collection.
   * @returns {Promise<void>}
   */
  async deleteAllDocuments(): Promise<void> {
    try {
      const querySnapshot = await this.firestore
        .collection(this.collectionName)
        .get();

      await this._processBatch(querySnapshot.docs, (batch, doc) =>
        batch.delete(doc.ref)
      );
    } catch (error) {
      this._handleError("delete all documents from Firestore", error);
    }
  }

  /**
   * Deletes documents by their IDs.
   * @param {string[]} ids - The IDs of the documents to delete.
   * @returns {Promise<void>}
   */
  async deleteDocumentsByIds(ids: string[]): Promise<void> {
    try {
      await this._processBatch(ids, (batch, id) => {
        const docRef = this.firestore.collection(this.collectionName).doc(id);
        batch.delete(docRef);
      });
    } catch (error) {
      this._handleError("delete documents by IDs from Firestore", error);
    }
  }

  /**
   * Deletes documents based on the provided filter.
   * @param {Record<string, any>} filter - The filter to apply.
   * @returns {Promise<void>}
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async deleteDocumentsByFilter(filter: Record<string, any>): Promise<void> {
    try {
      let query: FirebaseFirestore.Query<FirebaseFirestore.DocumentData> =
        this.firestore.collection(this.collectionName);

      for (const key of Object.keys(filter)) {
        const value = filter[key];
        const field = this._normalizeFilterField(key);
        query = query.where(field, "==", value);
      }

      const querySnapshot = await query.get();
      await this._processBatch(querySnapshot.docs, (batch, doc) =>
        batch.delete(doc.ref)
      );
    } catch (error) {
      this._handleError("delete documents by filter from Firestore", error);
    }
  }

  /**
   * Enhanced helper function for performing all types of similarity search operations.
   * @param {number[] | string} query - The query vector or text.
   * @param {number} k - The number of nearest neighbors to retrieve.
   * @param {Record<string, any>} [filter] - A filter to apply to the search.
   * @param {boolean} [withScores=false] - Whether to return scores with documents.
   * @param {boolean} [withEmbeddings=false] - Whether to include embeddings in the results.
   * @returns {Promise<DocumentInterface[] | Array<[DocumentInterface, number]> | Array<[DocumentInterface, number, number[]]>>} - The search results.
   */
  private async _similarity_search(
    query: number[] | string,
    k: number,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    filter?: Record<string, any>,
    withScores: boolean = false,
    withEmbeddings: boolean = false
  ): Promise<
    | DocumentInterface[]
    | Array<[DocumentInterface, number]>
    | Array<[DocumentInterface, number, number[]]>
  > {
    try {
      this._validateSearchParams(k, filter);
      const _filter = filter ?? this.filter;

      // Handle string queries by converting to vectors
      let queryVector: number[];
      if (typeof query === "string") {
        queryVector = await this.embeddings.embedQuery(query);
      } else {
        queryVector = query;
      }

      const coll = this.firestore.collection(this.collectionName);

      // Pre-filtering: Apply filters before vector search (requires composite vector index)
      let baseQuery: CollectionReference | Query = coll;
      if (_filter) {
        for (const key of Object.keys(_filter)) {
          baseQuery = baseQuery.where(key, "==", _filter[key]);
        }
      }

      const vectorQuery: VectorQuery = baseQuery.findNearest({
        vectorField: "embedding_field",
        queryVector,
        limit: k,
        distanceMeasure: this.distanceMeasure,
        distanceResultField: "vector_distance",
      });

      const querySnapshot: VectorQuerySnapshot = await vectorQuery.get();

      // Convert results to the desired format
      const results = querySnapshot.docs.map((doc) => {
        const data = doc.data();
        const document = this._createDocumentFromData(data);

        if (withEmbeddings) {
          const storedEmbedding = this._vectorToArray(data?.embedding_field);
          return [document, data.vector_distance, storedEmbedding] as [
            DocumentInterface,
            number,
            number[]
          ];
        } else {
          return [document, data.vector_distance] as [
            DocumentInterface,
            number
          ];
        }
      });

      if (withEmbeddings) {
        return results as Array<[DocumentInterface, number, number[]]>;
      } else if (withScores) {
        return results as Array<[DocumentInterface, number]>;
      } else {
        return results.map(([doc]) => doc) as DocumentInterface[];
      }
    } catch (error) {
      this._handleError("perform similarity search", error);
    }
  }

  /**
   * Performs a similarity search based on vector distance.
   * @param {number[]} query - The query vector.
   * @param {number} k - The number of nearest neighbors to retrieve.
   * @param {Object} [filter] - A filter to apply to the search.
   * @returns {Promise<Array<[DocumentInterface, number]>>} - The search results and their scores.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    filter?: Record<string, any>
  ): Promise<[DocumentInterface, number][]> {
    return this._similarity_search(query, k, filter, true) as Promise<
      [DocumentInterface, number][]
    >;
  }

  /**
   * Retrieves a document by its ID.
   * @param {string} id - The ID of the document to retrieve.
   * @returns {Promise<DocumentInterface | null>} - The retrieved document or null if not found.
   */
  async getDocumentById(id: string): Promise<DocumentInterface | null> {
    try {
      const docRef = this.firestore.collection(this.collectionName).doc(id);
      const doc = await docRef.get();
      if (!doc.exists) {
        return null;
      }
      const data: FirebaseFirestore.DocumentData | undefined = doc.data();
      return this._createDocumentFromData(data);
    } catch (error) {
      this._handleError("get document by ID from Firestore", error);
    }
  }

  /**
   * Retrieves documents by metadata filter.
   * @param {Record<string, any>} filter - The filter to apply to the search.
   * @returns {Promise<DocumentInterface[]>} - The retrieved documents.
   */
  async getDocumentsByMetadata(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    filter: Record<string, any>
  ): Promise<DocumentInterface[]> {
    try {
      let query: FirebaseFirestore.Query<FirebaseFirestore.DocumentData> =
        this.firestore.collection(this.collectionName);
      for (const key of Object.keys(filter)) {
        const value = filter[key];
        const field = this._normalizeFilterField(key);
        query = query.where(field, "==", value);
      }
      const querySnapshot = await query.get();
      return querySnapshot.docs.map((doc) => {
        const data = doc.data();
        return this._createDocumentFromData(data);
      });
    } catch (error) {
      this._handleError("get documents by metadata from Firestore", error);
    }
  }

  /**
   * Adds texts to the Firestore vector store.
   * @param {string[]} texts - The texts to add.
   * @param {AddTextsOptions} options - Options for adding texts.
   * @returns {Promise<string[]>} - Promise that resolves with the IDs of the added texts.
   */
  async addTexts(
    texts: string[],
    options?: AddTextsOptions
  ): Promise<string[]> {
    try {
      if (!texts.length) {
        return [];
      }

      const { metadatas = [], ids } = options || {};

      const documents = texts.map((text, index) => ({
        pageContent: text,
        metadata: Array.isArray(metadatas)
          ? metadatas[index] || {}
          : metadatas || {},
      }));

      return this.addDocuments(documents, { ids });
    } catch (error) {
      this._handleError("add texts to Firestore", error);
    }
  }

  /**
   * Performs a similarity search with the option to filter results.
   * @param {string} query - The query text to search for.
   * @param {number} k - The number of documents to return.
   * @param {SearchOptions} options - Search options including filters.
   * @returns {Promise<DocumentInterface[]>} - The search results.
   */
  async similaritySearch(
    query: string,
    kOrOptions?: number | SearchOptions,
    filter?: this["FilterType"],
    _callbacks?: Callbacks
  ): Promise<DocumentInterface[]> {
    // Handle both old and new parameter patterns for compatibility
    let k = 4;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let actualFilter;

    if (typeof kOrOptions === "number") {
      k = kOrOptions;
      actualFilter = typeof filter === "object" ? filter : undefined;
    } else if (typeof kOrOptions === "object" && kOrOptions !== null) {
      k = kOrOptions.k || 4;
      actualFilter =
        kOrOptions.filter || (typeof filter === "object" ? filter : undefined);
    }

    return this._similarity_search(query, k, actualFilter) as Promise<
      DocumentInterface[]
    >;
  }

  /**
   * Performs a similarity search using a vector.
   * @param {number[]} embedding - The embedding vector to search with.
   * @param {number} k - The number of documents to return.
   * @param {SearchOptions} options - Search options including filters.
   * @returns {Promise<DocumentInterface[]>} - The search results.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async similaritySearchByVector(
    embedding: number[],
    options?: SearchOptions,
    k = 4
  ): Promise<DocumentInterface[]> {
    return this._similarity_search(embedding, k, options?.filter) as Promise<
      DocumentInterface[]
    >;
  }

  /**
   * Performs a maximal marginal relevance search.
   * @param {string} query - The query text to search for.
   * @param {number} k - The number of documents to return.
   * @param {MaxMarginalRelevanceSearchOptions} options - Search options.
   * @returns {Promise<DocumentInterface[]>} - The search results.
   */
  async maxMarginalRelevanceSearchQuery(
    query: string,
    options?: MaxMarginalRelevanceSearchOptions,
    k = 4
  ): Promise<DocumentInterface[]> {
    try {
      const queryVector = await this.embeddings.embedQuery(query);
      return this.maxMarginalRelevanceSearchByVector(queryVector, options, k);
    } catch (error) {
      console.error(
        "Error performing maximal marginal relevance search:",
        error
      );
      throw new Error("Failed to perform maximal marginal relevance search.");
    }
  }

  /**
   * Performs a maximal marginal relevance search using a vector.
   * @param {number[]} embedding - The embedding vector to search with.
   * @param {number} k - The number of documents to return.
   * @param {MaxMarginalRelevanceSearchOptions} options - Search options.
   * @returns {Promise<DocumentInterface[]>} - The search results.
   */
  async maxMarginalRelevanceSearchByVector(
    embedding: number[],
    options?: MaxMarginalRelevanceSearchOptions,
    k = 4
  ): Promise<DocumentInterface[]> {
    try {
      const {
        fetchK = Math.max(k * 2, 20),
        lambdaMult = 0.5,
        filter,
      } = options || {};

      // Get more documents than needed for MMR calculation with embeddings
      const searchResultsWithEmbeddings = (await this._similarity_search(
        embedding,
        fetchK,
        filter,
        true,
        true
      )) as Array<[DocumentInterface, number, number[]]>;

      if (searchResultsWithEmbeddings.length === 0) {
        return [];
      }

      // Extract embeddings and documents from search results
      const candidateEmbeddings: number[][] = [];
      const candidateDocs: DocumentInterface[] = [];

      for (const [doc, , embeddingVector] of searchResultsWithEmbeddings) {
        candidateDocs.push(doc);

        if (embeddingVector && embeddingVector.length > 0) {
          candidateEmbeddings.push(embeddingVector);
        } else {
          // Fallback: re-embed the document text if embedding is missing or empty
          try {
            const docEmbedding = await this.embeddings.embedQuery(
              doc.pageContent
            );
            candidateEmbeddings.push(docEmbedding);
          } catch (error) {
            console.warn(`Failed to re-embed document ${doc.id}:`, error);
            // Skip this document if we can't get its embedding
            continue;
          }
        }
      }

      // If we still can't get embeddings, fall back to simple similarity search
      if (candidateEmbeddings.length === 0) {
        return searchResultsWithEmbeddings.slice(0, k).map(([doc]) => doc);
      }

      // Calculate MMR indices
      const mmrIndices = maximalMarginalRelevance(
        embedding,
        candidateEmbeddings,
        lambdaMult,
        k
      );

      return mmrIndices.map((index) => candidateDocs[index]);
    } catch (error) {
      console.error(
        "Error performing maximal marginal relevance search by vector:",
        error
      );
      throw new Error(
        "Failed to perform maximal marginal relevance search by vector."
      );
    }
  }

  /**
   * Utility method to convert Firestore Vector to array.
   * @param {any} vector - The Firestore vector field.
   * @returns {number[]} - The vector as an array.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _vectorToArray(vector: any): number[] {
    if (Array.isArray(vector)) {
      return vector;
    }
    if (vector && typeof vector === "object" && "_values" in vector) {
      return vector._values;
    }
    if (vector && typeof vector === "object" && "values" in vector) {
      return vector.values;
    }
    return [];
  }

  /**
   * Static method that creates a new instance of the FirestoreVectorStore class from documents.
   * @param {DocumentInterface[]} docs - The documents to add to the Firestore vector database.
   * @param {EmbeddingsInterface} embeddings - The embeddings to use for the documents.
   * @param {FirebaseStoreParams} params - The parameters for the Firestore configuration.
   * @returns {Promise<FirestoreVectorStore>} - Promise that resolves with a new instance of the class.
   */
  static async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface,
    params: FirebaseStoreParams
  ): Promise<FirestoreVectorStore> {
    const instance = new FirestoreVectorStore({ embeddings, params });
    await instance.addDocuments(docs);
    return instance;
  }

  /**
   * Static method that creates a new instance of the FirestoreVectorStore class from texts.
   * @param {string[]} texts - The texts to add to the Firestore database.
   * @param {object | object[]} metadatas - Metadata associated with the texts.
   * @param {EmbeddingsInterface} embeddings - The embeddings to use for the texts.
   * @param {FirebaseStoreParams} params - The parameters for the Firestore configuration.
   * @returns {Promise<FirestoreVectorStore>} - Promise that resolves with a new instance of the class.
   */
  static async fromTexts(
    texts: string[],
    metadatas: object | object[],
    embeddings: EmbeddingsInterface,
    params: FirebaseStoreParams
  ): Promise<FirestoreVectorStore> {
    if (!texts.length) {
      return new FirestoreVectorStore({ embeddings, params });
    }

    const docs = texts.map((text, index) => {
      const metadata = Array.isArray(metadatas) ? metadatas[index] : metadatas;
      return new Document({
        pageContent: text,
        metadata: metadata || {},
      });
    });
    return FirestoreVectorStore.fromDocuments(docs, embeddings, params);
  }
}

export { FirestoreVectorStore };
