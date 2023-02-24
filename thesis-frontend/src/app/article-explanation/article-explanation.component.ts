import { Component,Input,ViewChild } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';

@Component({
  selector: 'app-article-explanation',
  templateUrl: './article-explanation.component.html',
  styleUrls: ['./article-explanation.component.css']
})
export class ArticleExplanationComponent {
  @Input() clickedPoint: any;
  @Input() data: any[];
  @ViewChild(BaseChartDirective) chart: BaseChartDirective;

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
    }
  };

  ngOnChanges(): void {
    const requiredData = this.data.filter(d=>d.article_no===this.clickedPoint);
    if (this.data) {
      this.chartData.datasets = [{data: Object.values(Object.assign({}, ...requiredData)).slice(0,requiredData.length - 5),
        borderWidth: 1,
        label: 'Article Explanation ' + this.clickedPoint,
        barPercentage: 0.5,
        barThickness: 6,
        maxBarThickness: 8,
        minBarLength: 2,
      }];
      this.chartData.labels = [...Array(Object.values(Object.assign({}, ...requiredData)).slice(0,requiredData.length - 5).length).keys()];
    }
  this.chart.chart?.update();
  }
}
