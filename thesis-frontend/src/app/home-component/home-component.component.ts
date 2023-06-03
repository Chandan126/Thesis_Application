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
  dataSources: any;
  systemOption: any = 'Select System';
  source: any;
  labels: any;
  clusterNumbers: any = [];
  data: any;
  globalExplanations: any;
  globalExplanationsA: any;
  globalExplanationsB: any;
  selectedExplanationType: string = 'local';
  articles: any;
  selectedGlobalCluster: any;
  article1: any;
  article2: any;
  articleFeatureDiv: any;
  loadingGlobalExplanations: boolean = false;
  query: any;
  featureWeight: any[] = [2,2,2,2,2];
  isArticleExplanation = false;
  interestingClusters:any[] = [];
  notInterestingClusters:any[] = [];
  configuration: string | undefined;
  clickedPoint: any;
  localExplanations: any;
  explanation_dd: boolean = false;
  user_feedback_dd: boolean = false;

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
    }, 3000);
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
      //this.dataService.getGlobalExplanations(this.source),
      this.dataService.getFeatureDivision(this.source)
    ).subscribe(([response1, response2, response3]) => {
      this.labels = response1;
      this.clusterNumbers = Array.from({length: this.labels}, (_, i) => `Cluster ${i}`);
      this.data = response2;
      //this.globalExplanations = response3;
      this.articleFeatureDiv = response3;
      this.spinner.hide();
    });
  }

  getGlobalExplanation(){
    if(this.selectedExplanationType=='global' && this.systemOption!='Select System'){
      if(this.globalExplanations && this.globalExplanationsA && this.systemOption=='A'){
        this.globalExplanations = this.globalExplanationsA;
        return;
      }
      else if(this.globalExplanations && this.globalExplanationsB && this.systemOption=='B'){
        this.globalExplanations = this.globalExplanationsB;
        return;
      }
      console.log('Global Explanation on the way');
      this.loadingGlobalExplanations = true;
      this.dataService.getGlobalExplanations(this.source,this.systemOption)
      .subscribe(response => {
        //console.log(response);
        this.globalExplanations = response;
        this.loadingGlobalExplanations = false;
        if(this.systemOption=='A'){
          this.globalExplanationsA = response;
        }
        else if(this.systemOption=='B'){
          this.globalExplanationsB = response;
        }
      });
    }
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

  handleExpButtonClick(){
    this.explanation_dd = !this.explanation_dd;
    this.user_feedback_dd = false
  }

  handleUserFeedButtonClick(){
    this.user_feedback_dd = !this.user_feedback_dd;
    this.explanation_dd = false
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
    this.globalExplanationsA = undefined;
    this.globalExplanationsB = undefined;
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
