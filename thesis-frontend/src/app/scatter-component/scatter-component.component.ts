import { Component, Input, Output, ViewChild, EventEmitter   } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { MatDialog, MatDialogRef  } from '@angular/material/dialog';
import {
  ActiveElement,
  ChartEvent,
} from 'chart.js';
import {FacetExplanationComponentComponent} from '../facet-explanation-component/facet-explanation-component.component';
import { NgZone } from '@angular/core';


@Component({
  selector: 'app-scatter-component',
  templateUrl: './scatter-component.component.html',
  styleUrls: ['./scatter-component.component.css']
})
export class ScatterComponentComponent {
  @Input() data: any[];
  @Input() labels: any;
  @Input() articleFeatureDiv: any;
  @Input() source: any;
  @Output() public clusterFeedback: EventEmitter<any> = new EventEmitter<any>();
  @ViewChild(BaseChartDirective) chart: BaseChartDirective;
  dialogRef: MatDialogRef<FacetExplanationComponentComponent>;
  
  constructor(public dialog: MatDialog, private zone: NgZone) {
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
        this.zone.run(() => {
          this.dialogRef  = this.dialog.open(FacetExplanationComponentComponent, {
            position: { top: '5px', left: '5px' },
            data: { selected_article: selected_article, articleFeatureDiv: this.articleFeatureDiv, data: this.data, source: this.source}
        })

          this.dialogRef.afterClosed().subscribe(result => {
            //console.log('The dialog was closed', result);
            this.zone.run(() => this.clusterFeedback.emit(result));
          });
        }
      );
      }
    }
  };

  ngOnChanges(): void {
    const colours = ['blue','red','orange','green','yellow','purple','pink','black','turquoise','crimson'];
    const generateDatasets: any = [];
    for(let i=0; i<parseInt(this.labels); i++){
      generateDatasets.push({data: this.data.filter(d => d.k_labels == i).map((d) => {
        return { x: d['x_axis'], y: d['y_axis'], article_no: d['article_no'],highlight: d['highlight'] == 1 ? true : false };
          }),
          label: i,
          pointRadius: (d: any) => d.raw.highlight ? 15 : 5,
          backgroundColor: colours[i],
          borderColor: (d:any) => d.raw.highlight ? 'black' : 'white'
        })
    }
    if (this.data) {
      this.chartData.datasets = generateDatasets;
    }
  this.chart?.chart?.update();
  }

}
