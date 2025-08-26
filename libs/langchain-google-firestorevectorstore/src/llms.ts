// eslint-disable-next-line import/no-extraneous-dependencies
import { Firestore, FieldValue } from "@google-cloud/firestore";
// eslint-disable-next-line import/no-extraneous-dependencies
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { DocumentInterface } from "@langchain/core/documents";
import { VectorStore } from "@langchain/core/vectorstores";
import {
  AsyncCaller,
  AsyncCallerParams,
} from "@langchain/core/utils/async_caller";
// import { Compute } from "google-auth-library";
import { GoogleAuth } from "google-auth-library";

export interface FirebaseStoreParamsSimple extends AsyncCallerParams {
  // firestore: Firestore;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  firestoreConfig?: FirebaseFirestore.Settings;
  GoogleAuth: GoogleAuth;
}

class FirestoreVectorStoreSimpleAuth extends VectorStore {
  firestore: Firestore;

  caller: AsyncCaller;

  firestoreConfig: FirebaseFirestore.Settings;

  constructor({
    embeddings,
    params,
  }: {
    embeddings: EmbeddingsInterface;
    params: FirebaseStoreParamsSimple;
  }) {
    super(embeddings, params);
    const { GoogleAuth, firestoreConfig, ...asyncCallerArgs } = params;
    this.caller = new AsyncCaller(asyncCallerArgs);
    this.firestore = new Firestore({ GoogleAuth, firestoreConfig });
  }

  _vectorstoreType(): string {
    return "FirestoreVectorStore";
  }

  async addVectors(
    _vectors: number[][],
    _documents: DocumentInterface<Record<string, any>>[],
    _options?: { [x: string]: any } | undefined
  ): Promise<void | string[]> {
    const coll = this.firestore.collection("coffee-beans-2");
    await coll.add({
      name: "Ramesh Krishna coffee Using Auth in LLM ",
      description: "Information about the Kahawa coffee beans.",
      embedding_field: FieldValue.vector([1.0, 2.0, 3.0]),
    });
    console.log("Document added");
  }

  addDocuments(
    _documents: DocumentInterface<Record<string, any>>[],
    _options?: { [x: string]: any } | undefined
  ): Promise<void | string[]> {
    throw new Error("Method not implemented.");
  }

  similaritySearchVectorWithScore(
    _query: number[],
    _k: number,
    _filter?: this["FilterType"] | undefined
  ): Promise<[DocumentInterface<Record<string, any>>, number][]> {
    throw new Error("Method not implemented.");
  }
}

export { FirestoreVectorStoreSimpleAuth };
