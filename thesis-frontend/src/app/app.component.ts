import { Component, OnInit } from '@angular/core';
import {DataServiceService} from '../app/data-service.service'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'thesis-frontend';
  sessionId: any;

  constructor(private dataService: DataServiceService){
  }

  ngOnInit() {
    this.dataService.initSession();
  }
}
