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
}
