import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataServiceService {

  constructor(private http: HttpClient) { }

  getSourceData() {
    return this.http.get('http://127.0.0.1:8000/sources');
  }

  getData(source: string){
    return this.http.get(`http://127.0.0.1:8000/sources/${source}`)
  }

  getDataLabels(source: string){
    return this.http.get(`http://127.0.0.1:8000/labels/${source}`)
  }

  getGlobalExplanations(source: string){
    return this.http.get(`http://127.0.0.1:8000/global_explanations/${source}`)
  }
}
