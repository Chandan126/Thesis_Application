import { Component,ChangeDetectorRef, OnInit } from '@angular/core';
import { DataServiceService } from '../data-service.service';
import {Router} from '@angular/router'
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
  featureWeight: any[] = [2,2,2,2,2];
  isArticleExplanation = false;
  interestingClusters:any[] = [];
  notInterestingClusters:any[] = [];
  configuration: string | undefined;
  clickedPoint: any;
  localExplanations: any;

  constructor(private dataService: DataServiceService,
    private changeDetectorRef: ChangeDetectorRef,
    private spinner: NgxSpinnerService,private router: Router
    ) {}

  ngOnInit() {
    setTimeout(() => {
      // Code to execute after 5 seconds
      const sessionId = this.dataService.getSessionId();
      if(sessionId){
        this.getDataSourceFromAPI();
      }
      else{
        this.router.navigateByUrl('/login');
      }
    }, 5000);
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
    //this.getDataFromSource({target:{value:this.source}},this.query);
    this.dataService.getData(this.source,this.query).subscribe(response => this.data = response)
  }

  onFeatureReweighting(value: any):void {
    console.log(value);
    this.featureWeight = value;
  }

  recluster(){
    const result = this.dataService.getUserFeedbacks();
    //this.query = '';
    console.log(result);
    this.spinner.show();
    this.configuration = undefined;
    this.dataService.reClusterWords(this.articleFeatureDiv,this.source,this.featureWeight,result['interestingClusters'],result['notInterestingClusters'],result['relevantDocs'],result['notRelevantDocs']).subscribe(
      result => {
      console.log('The dialog was closed', result);
      this.spinner.hide();
      this.getDataFromSource({target:{value:this.source}},undefined);
    });
    //console.log(this.featureWeight);
  }
}
