import { Component,Input,ViewChild, Output,EventEmitter } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { MatDialog } from '@angular/material/dialog';

import {
  ActiveElement,
  ChartEvent,
} from 'chart.js';
import { NgZone } from '@angular/core';

@Component({
  selector: 'app-article-explanation',
  templateUrl: './article-explanation.component.html',
  styleUrls: ['./article-explanation.component.css']
})
export class ArticleExplanationComponent {
  @Input() clickedPoint: any;
  @Input() data: any[];
  @Input() articleFeatureDiv: any;
  @Output() public barClicked: EventEmitter<any> = new EventEmitter<any>();
  @ViewChild(BaseChartDirective) chart: BaseChartDirective;
  constructor(public dialog: MatDialog, private zone: NgZone) {}

  chartData: any = {
    datasets: [{
      label: 'Bar Dataset',
      data: [0,0,0,0]
    }],
  };
  options: any = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    scales: {
      x: {
        grid: {
          offset: true
        }
    },
      y: {
        beginAtZero: true
      }
    },
    onClick: (
      event: ChartEvent,
      elements: ActiveElement[]
    ) => {
      if (elements[0]) {
          const selected_article = this.chartData.labels[elements[0].index];
          this.zone.run(() => this.barClicked.emit(selected_article));
      }
    },
  };

  ngOnChanges(): void {
    const requiredData = this.data.filter(d=>d.article_no===this.clickedPoint);
    console.log(requiredData);
    console.log(this.articleFeatureDiv);
    if (this.data) {
      this.chartData.datasets = [{data: Object.values(Object.assign({}, ...requiredData)).slice(0,requiredData.length - 5),
        borderWidth: 1,
        label: 'Article Explanation ' + this.clickedPoint,
        barPercentage: 0.5,
        barThickness: 6,
        maxBarThickness: 8,
        minBarLength: 2,
      }];
      console.log(this.data);
      this.chartData.labels = this.articleFeatureDiv;
    }
  this.chart.chart?.update();
  }
}
