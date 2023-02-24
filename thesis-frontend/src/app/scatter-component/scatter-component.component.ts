import { Component, Input, Output, ViewChild, EventEmitter   } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import {
  ActiveElement,
  ChartEvent,
} from 'chart.js';

@Component({
  selector: 'app-scatter-component',
  templateUrl: './scatter-component.component.html',
  styleUrls: ['./scatter-component.component.css']
})
export class ScatterComponentComponent {
  @Input() data: any[];
  @Input() labels: any;
  @Output() isArticleExplanationChanged = new EventEmitter<boolean>();
  @Output() pointClicked = new EventEmitter<any>();

  @ViewChild(BaseChartDirective) chart: BaseChartDirective;
  constructor() {
    this.data = [];
  }

  chartData: any = {
    datasets: [{
      label: 'Scatter Dataset',
      pointRadius: 5,
      borderColor: 'blue',
      backgroundColor: 'transparent',
      data: []
    }],
  };
  options: any = {
    scales: {
      x: {
        type: 'linear',
        position: 'bottom'
      }
    },
    plugins: {
      tooltip: {
          callbacks: {
              label: function(tooltipItem:any) {
                const articleNo = tooltipItem.dataset.data[tooltipItem.dataIndex].article_no;
                return `Article No: ${articleNo}`;
              }
          }
      }
    },
    onClick: (
      event: ChartEvent,
      elements: ActiveElement[]
    ) => {
      if (elements[0]) {
        const selected_article = this.chartData.datasets[elements[0].datasetIndex].data[elements[0].index].article_no;
        this.pointClicked.emit(selected_article);
        this.isArticleExplanationChanged.emit(true);
        console.log('Clicked on ', selected_article);
      }
    },
  };
  ngOnChanges(): void {
    const colours = ['blue','red','orange','green','yellow'];
    const generateDatasets: any = [];
    for(let i=0; i<parseInt(this.labels); i++){
      generateDatasets.push({data: this.data.filter(d => d.k_labels == i).map((d) => {
        return { x: d['x_axis'], y: d['y_axis'], article_no: d['article_no'] };
          }),
          label: i,
          pointRadius: 5,
          backgroundColor: colours[i],
        })
    }
    if (this.data) {
      this.chartData.datasets = generateDatasets;
    }
  this.chart.chart?.update();
  }

}
