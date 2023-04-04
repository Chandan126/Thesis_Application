import { Component, Inject, OnInit, NgZone, Output, EventEmitter } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import {DataServiceService} from '../data-service.service';
import { MatDialogRef } from '@angular/material/dialog';

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
    console.log(value);
    this.clickedWord = value;
    this.isWordClick = true;
  }

  onClusterFeedback(value: any) {
    //this.ngZone.run(() => this.clusterFeedback.emit(value));
    this.ngZone.run(() => this.dialogRef.close(value));
  }

}
