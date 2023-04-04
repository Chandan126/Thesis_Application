import { Component,Input, Output,EventEmitter } from '@angular/core';
import {DataServiceService} from '../data-service.service';

@Component({
  selector: 'app-word-reassign-component',
  templateUrl: './word-reassign-component.component.html',
  styleUrls: ['./word-reassign-component.component.css']
})
export class WordReassignComponentComponent{
  @Input() clickedWord: any;
  @Input() source: any;
  @Input() articleFeatureDiv: any;
  @Input() clickedBar:any;
  //@Output() public clusterOpinion: EventEmitter<any> = new EventEmitter<any>();
  selectedCluster: any;
  filteredArticleFeatureDiv: any;
  similarWords: any;

  constructor(private dataService: DataServiceService){
  }

  ngOnChanges() {
    this.filterArticleFeatureDiv();
  }

  getWords(){
    this.dataService.getSimilarWords(this.source,this.selectedCluster,this.clickedWord)
    .subscribe(data => this.similarWords = data);
    console.log(this.similarWords);
  }

  filterArticleFeatureDiv(){
    this.filteredArticleFeatureDiv = this.articleFeatureDiv.filter((cluster: any) => {
      return cluster.includes(this.clickedBar.split(' ')[0]);
    });
  }

  onConfirmClick() {
      console.log(`User wants to move ${this.clickedWord} from ${this.clickedBar} to ${this.selectedCluster}`)
  }
  
}
