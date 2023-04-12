import { Component,ChangeDetectorRef, OnInit } from '@angular/core';
import { DataServiceService } from '../data-service.service';
import { NgxSpinnerService } from 'ngx-spinner';
import { zip } from 'rxjs';

@Component({
  selector: 'app-home-component',
  templateUrl: './home-component.component.html',
  styleUrls: ['./home-component.component.css']
})
export class HomeComponentComponent implements OnInit {
  dataSources: any = [];
  source: any;
  labels: any;
  clusterNumbers: any = [];
  data: any;
  globalExplanations: any;
  selectedExplanationType: string = 'global';
  articles: any;
  selectedGlobalCluster: any;
  article1: any;
  article2: any;
  articleFeatureDiv: any;
  query: any;
  featureWeight: any[] = [1,1,1,1,1];
  isArticleExplanation = false;
  interestingClusters:any[] = [];
  notInterestingClusters:any[] = [];
  configuration: string | undefined;
  clickedPoint: any;
  localExplanations: any;

  constructor(private dataService: DataServiceService,
    private changeDetectorRef: ChangeDetectorRef,
    private spinner: NgxSpinnerService
    ) {}

  ngOnInit() {
    this.getDataSourceFromAPI();
  }

  getDataSourceFromAPI() {
    this.dataService.getSourceData().subscribe(data => {
      this.dataSources = data;
    });
  }

  getDataFromSource(event: any,query: string | undefined){
    this.spinner.show();
    this.source = event.target.value;
    zip(
      this.dataService.getDataLabels(this.source),
      this.dataService.getData(this.source,query),
      this.dataService.getGlobalExplanations(this.source),
      this.dataService.getFeatureDivision(this.source)
    ).subscribe(([response1, response2, response3, response4]) => {
      this.labels = response1;
      this.clusterNumbers = Array.from({length: this.labels}, (_, i) => `Cluster ${i}`);
      this.data = response2;
      this.globalExplanations = response3;
      this.articleFeatureDiv = response4;
      this.spinner.hide();
    });
  }

  selectedClusterNumberChange(){
    this.dataService.getArticlesForCluster(this.source,this.selectedGlobalCluster)
    .subscribe(articles => this.articles = articles);
  }

  onDropdownChange() {
    if (this.article1 && this.article2) {
      this.dataService.getLocalExplanations(this.source,this.article1,this.article2)
      .subscribe(result => this.localExplanations = result);
    }
  }

  onPointClicked(point: any) {
    this.clickedPoint = point;
    this.changeDetectorRef.detectChanges();
  }

  handleButtonClick(input: string){
    this.configuration = input
  }

  search(){
    console.log(this.query);
    this.getDataFromSource({target:{value:this.source}},this.query);
  }

  onFeatureReweighting(value: any):void {
    console.log(value);
    this.featureWeight = value;
  }

  onClusterFeedback(value: any) {
    this.interestingClusters.push(value['interestingClusters']);
    this.notInterestingClusters.push(value['notInterestingClusters']);
    console.log(`Cluster Feedback is ${this.interestingClusters}`);
    console.log(`Cluster Feedback is ${this.notInterestingClusters}`);
  }

  recluster(){
    this.spinner.show();
    this.configuration = undefined;
    const interestingClusters = this.interestingClusters.filter((val) => val !== undefined);
    const notInterestingClusters = this.notInterestingClusters.filter((val) => val !== undefined);
    this.dataService.reClusterWords(this.articleFeatureDiv,this.source,this.featureWeight,interestingClusters,notInterestingClusters).subscribe(
      result => {
      console.log('The dialog was closed', result);
      this.spinner.hide();
      this.getDataFromSource({target:{value:this.source}},undefined);
    });
    //console.log(this.featureWeight);
  }
}
