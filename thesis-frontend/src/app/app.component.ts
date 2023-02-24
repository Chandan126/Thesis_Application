import { Component, ChangeDetectorRef } from '@angular/core';
import {DataServiceService} from './data-service.service';
import { zip } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'thesis-frontend';
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
  isArticleExplanation = false;
  clickedPoint: any;
  localExplanations: any;

  constructor(private dataService: DataServiceService,private changeDetectorRef: ChangeDetectorRef) {}

  ngOnInit() {
    this.getDataSourceFromAPI();
  }

  getDataSourceFromAPI() {
    this.dataService.getSourceData().subscribe(data => {
      this.dataSources = data;
    });
  }

  getDataFromSource(event: any){
    this.source = event.target.value;
    zip(
      this.dataService.getDataLabels(this.source),
      this.dataService.getData(this.source),
      this.dataService.getGlobalExplanations(this.source)
    ).subscribe(([response1, response2, response3]) => {
      this.labels = response1;
      this.clusterNumbers = Array.from({length: this.labels}, (_, i) => `Cluster ${i}`);
      this.data = response2;
      this.globalExplanations = response3;
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

  onIsArticleExplanationChanged(value: boolean) {
    this.isArticleExplanation = value;
    this.changeDetectorRef.detectChanges();
  }

  onPointClicked(point: any) {
    this.clickedPoint = point;
    this.changeDetectorRef.detectChanges();
  }
}
