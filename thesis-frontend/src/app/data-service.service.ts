import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataServiceService {

  private sessionId: any;
  constructor(private http: HttpClient) { }

  initSession(){
    return this.http.get('http://127.0.0.1:8000/session').subscribe((data) => {
      this.sessionId = data;
      console.log(`Session started ${this.sessionId.sessionId}`);
    });;
  }
  getSourceData() {
    return this.http.get('http://127.0.0.1:8000/sources');
  }

  getData(source: string){
    return this.http.get(`http://127.0.0.1:8000/sources/${this.sessionId.sessionId}/${source}`)
  }

  getDataLabels(source: string){
    return this.http.get(`http://127.0.0.1:8000/labels/${source}`)
  }

  getGlobalExplanations(source: string){
    return this.http.get(`http://127.0.0.1:8000/global_explanations/${this.sessionId.sessionId}/${source}`)
  }

  getArticlesForCluster(source: string,selectedClusterNumber: string){
    return this.http.get(`http://127.0.0.1:8000/get_articles/${this.sessionId.sessionId}/${source}/${selectedClusterNumber}`)
  }

  getLocalExplanations(source: string,article1: string,article2: string){
    return this.http.get(`http://127.0.0.1:8000/local_explanations/${this.sessionId.sessionId}/${source}/${article1}/${article2}`)
  }

  getFeatureDivision(source: string){
    return this.http.get(`http://127.0.0.1:8000/get_article_div/${this.sessionId.sessionId}/${source}`)
  }

  getArticleContent(source: string,article: string){
    return this.http.get(`http://127.0.0.1:8000/get_article_content/${this.sessionId.sessionId}/${source}/${article}`)
  }

  getFacetExplanation(source: string,facet: string,article_no:string){
    return this.http.get(`http://127.0.0.1:8000/get_facet_explanation/${this.sessionId.sessionId}/${source}/${facet}/${article_no}`)
  }

  getSimilarWords(source: string,facet: string,word:string){
    return this.http.get(`http://127.0.0.1:8000/get_similar_words/${this.sessionId.sessionId}/${source}/${facet}/${word}`)
  }

  reassignWords(source: string,facet: string,word:string,new_cluster:string){
    return this.http.get(`http://127.0.0.1:8000/reassign_words/${this.sessionId.sessionId}/${source}/${facet}/${word}/${new_cluster}`)
  }
}
