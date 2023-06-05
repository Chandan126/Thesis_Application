import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class DataServiceService {

  private sessionId: any;
  relevantDocs: any[] = [];
  selectedSystem: string;
  notRelevantDocs: any[] = [];
  interestingClusters: any[] = [];
  notinterestingClusters: any[] = [];
  constructor(private http: HttpClient) { }

  initSession(system: string){
    this.selectedSystem = system;
    return this.http.get(`http://127.0.0.1:8000/session/${system}`).subscribe((data) => {
      this.sessionId = data;
      localStorage.setItem('sessionId', this.sessionId.sessionId);
      console.log(`Session started ${this.sessionId.sessionId}`);
    });
  }

  getSessionId(){
    return this.sessionId?.sessionId;
  }

  storeClusterFeedback(doc: any[], interesting: boolean) {
    if(interesting==true){
      this.interestingClusters.push(doc);
    }
    else{
      this.notinterestingClusters.push(doc);
    }
    console.log(`Rel Clus are ${this.interestingClusters} and Not Rel Clus are ${this.notinterestingClusters}`)
  }

  storeDocsFeedback(doc: any,relevance: boolean) {
    if(relevance==true){
      this.relevantDocs.push(doc);
    }
    else{
      this.notRelevantDocs.push(doc);
    }
    console.log(`Rel Docs are ${this.relevantDocs} and Not Rel Docs are ${this.notRelevantDocs}`)
  }

  getUserFeedbacks(){
    console.log(this.selectedSystem);
    return {'relevantDocs':this.relevantDocs,'notRelevantDocs':this.notRelevantDocs,
    'interestingClusters':this.interestingClusters, 'notInterestingClusters':this.notinterestingClusters  }
  }
  getSourceData() {
    return this.http.get('http://127.0.0.1:8000/sources');
  }

  getData(source: string,query: string|undefined){
    return this.http.get(`http://127.0.0.1:8000/sources/${this.sessionId.sessionId}/${source}/${query}`);
  }

  getDataLabels(source: string){
    return this.http.get(`http://127.0.0.1:8000/labels/${source}`);
  }

  getGlobalExplanations(source: string,system: string){
    return this.http.get(`http://127.0.0.1:8000/global_explanations/${this.sessionId.sessionId}/${source}/${system}`);
  }

  getArticlesForCluster(source: string,selectedClusterNumber: string){
    return this.http.get(`http://127.0.0.1:8000/get_articles/${this.sessionId.sessionId}/${source}/${selectedClusterNumber}`);
  }

  getLocalExplanations(source: string,article1: string,article2: string){
    return this.http.get(`http://127.0.0.1:8000/local_explanations/${this.sessionId.sessionId}/${source}/${article1}/${article2}`);
  }

  getFeatureDivision(source: string){
    return this.http.get(`http://127.0.0.1:8000/get_article_div/${this.sessionId.sessionId}/${source}`);
  }

  getArticleContent(source: string,article: string){
    return this.http.get(`http://127.0.0.1:8000/get_article_content/${this.sessionId.sessionId}/${source}/${article}`);
  }

  getFacetExplanation(source: string,facet: string,article_no:string){
    return this.http.get(`http://127.0.0.1:8000/get_facet_explanation/${this.sessionId.sessionId}/${this.selectedSystem}/${source}/${facet}/${article_no}`);
  }

  getSimilarWords(source: string,facet: string,word:string){
    return this.http.get(`http://127.0.0.1:8000/get_similar_words/${this.sessionId.sessionId}/${source}/${facet}/${word}`);
  }

  reassignWords(source: string,facet: string,word:string,new_cluster:string){
    return this.http.get(`http://127.0.0.1:8000/reassign_words/${this.sessionId.sessionId}/${source}/${facet}/${word}/${new_cluster}`);
  }

  logout(){
    return this.http.get(`http://127.0.0.1:8000/logout/${this.sessionId.sessionId}`);
  }

  reClusterWords(feature_sizes_k: string[],source:string,global_weights: string[],increase_local_weights: string[],decrease_local_weights:string[],relevant_docs:string[],not_relevant_docs:string[]){
    const feature_sizes_k_json = JSON.stringify(feature_sizes_k);
    const global_weights_json = JSON.stringify(global_weights);
    const relevant_docs_json = JSON.stringify(relevant_docs);
    const not_relevant_docs_json = JSON.stringify(not_relevant_docs);
    const increase_local_weights_json = JSON.stringify(increase_local_weights) ? JSON.stringify(increase_local_weights) : null;
    const decrease_local_weights_json = JSON.stringify(decrease_local_weights) ? JSON.stringify(decrease_local_weights) : null;
    return this.http.get(`http://127.0.0.1:8000/recluster_words/${this.sessionId.sessionId}/${this.selectedSystem}/${source}/${feature_sizes_k_json}/${relevant_docs_json}/${not_relevant_docs_json}/${global_weights_json}/${increase_local_weights_json}/${decrease_local_weights_json}`);
  }
}
