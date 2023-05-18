import { Component, Inject, OnInit, NgZone, Output, EventEmitter } from '@angular/core';
import { MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import {DataServiceService} from '../data-service.service';

@Component({
  selector: 'app-facet-explanation-component',
  templateUrl: './facet-explanation-component.component.html',
  styleUrls: ['./facet-explanation-component.component.css']
})
export class FacetExplanationComponentComponent implements OnInit{

  content: any;
  isContent = true;
  isWordClick = false;
  clickedBar: any;
  clickedWord: any;
  //@Output() public clusterFeedback: EventEmitter<any> = new EventEmitter<any>();

  constructor(@Inject(MAT_DIALOG_DATA) public data: any,private dataService: DataServiceService, private dialogRef: MatDialogRef<FacetExplanationComponentComponent>, private ngZone: NgZone) { 
    this.content = {
      'article_title': null,
      'article_data': null
    }
  }

  ngOnInit(): void {
    this.dataService.getArticleContent(this.data.source,this.data.selected_article).subscribe(data => {
      this.content = data;
    });
  }

  onBarClicked(value: any): void {
    this.clickedBar = value;
    this.isContent = false;
  }

  onWordClicked(value: any):void {
    this.clickedWord = value;
    this.isWordClick = true;
  }

  onRelevanceClick(){
    this.dataService.storeDocsFeedback(this.data.selected_article, true);
  }

  onNotRelevanceClick(){
    this.dataService.storeDocsFeedback(this.data.selected_article, false);
  }

  close() {
    this.dialogRef.close();
  }

}
