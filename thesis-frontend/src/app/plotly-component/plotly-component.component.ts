import { Component, Input, NgZone, ViewChild } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { FacetExplanationComponentComponent } from '../facet-explanation-component/facet-explanation-component.component';
import { PlotlyComponent} from 'angular-plotly.js';

@Component({
  selector: 'app-plotly-component',
  templateUrl: './plotly-component.component.html',
  styleUrls: ['./plotly-component.component.css']
})
export class PlotlyComponentComponent {

  @Input() data: any[];
  @Input() labels: any;
  @Input() articleFeatureDiv: any;
  @Input() source: any;

  @ViewChild('plotlyInstance') plotlyInstance: any;
  @ViewChild(PlotlyComponent) plotlyComponent: PlotlyComponent;

  dialogRef: MatDialogRef<FacetExplanationComponentComponent>;

  graph_data: any[];

  showPlot = false;

  constructor(public dialog: MatDialog, private zone: NgZone) {
    this.data = [];
    console.log(this.data);
  }


  public graph_layout = {
    title: 'Articles',
    width: 1500, // Width of the graph
    height: 750, // Height of the graph
    xaxis: {
      showline: false, // hide the x-axis line
      showticklabels: true, // show x-axis labels
      zeroline: false // hide the thick line at y=0
    },
    yaxis: {
      showline: false, // hide the y-axis line
      showticklabels: true, // show y-axis labels
      zeroline: false // hide the thick line at x=0
    }
  };

  // change the font-size for highlighting some specific labels
  generateMarker(highlight: number, relevance: number) {
    return {
      size: (highlight == 1 && relevance == 1) ? 30 : 12,
      opacity: (highlight == 1 && relevance == 1) ? 1 : 0.6,
      line: {
        color: (highlight == 1 && relevance == 1) ? 'black' : 'white',
      },
    };
  }

  // generate traces for the plotly.
  generateTraces(): void {
    if (this.data && this.data.length) {
      let traces: any = {};
      
      // Loop through the data
      for (let item of this.data) {
        let k_label = item.k_labels;

        // Initialize trace if not present
        if (!traces[k_label]) {
          traces[k_label] = {
            x: [],
            y: [],
            mode: 'markers',
            type: 'scatter',
            name: `cluster-label ${k_label}`,
            text: [],
            marker: {
              size: [],
              opacity: [],
              line: {
                color: []
              }
            }
          };
        }

        // Add data to trace
        let marker = this.generateMarker(item.highlight, item.relevance);
        traces[k_label].x.push(item.x_axis);
        traces[k_label].y.push(item.y_axis);
        traces[k_label].text.push(`article_no-${item.article_no}`);
        traces[k_label].marker.size.push(marker.size);
        traces[k_label].marker.opacity.push(marker.opacity);
        traces[k_label].marker.line.color.push(marker.line.color);
      }

      // Convert object to array
      this.graph_data = Object.values(traces);

      console.log(this.graph_data);

      this.showPlot = true;

      this.assignClickEvent();      
    } else {
      console.log('Data is either null, undefined or empty');
    }
  }


// assign a on-click event when a specific point is being clicked.
assignClickEvent() {
  if (this.plotlyComponent && this.plotlyComponent.plotlyInstance) {
    this.plotlyComponent.plotlyInstance.on('plotly_click', (data: any) => {
      data.event.stopPropagation();
      let clickedPoints: number[] = []; // Track the clicked points

      // Loop through the clicked points
      for (let point of data.points) {
        // Get the text metadata of the clicked point
        // split the string into two parts: ["article_no", "3354"]
        let parts = point.data.text[point.pointNumber].split("-");
        // parse the second part into a number
        let article_num = parseInt(parts[1]);

        // Add the clicked point to the array
        clickedPoints.push(article_num);
      }

      // Open the dialog for the last clicked point
      if (clickedPoints.length > 0) {
        let article_num = clickedPoints[clickedPoints.length - 1];

        this.zone.run(() => {
          this.dialogRef = this.dialog.open(FacetExplanationComponentComponent, {
            width: '1500px', height: '700px',
            data: { selected_article: article_num, articleFeatureDiv: this.articleFeatureDiv, data: this.data, source: this.source }
          });
        });
      }
    });
  }
}

  ngOnChanges(): void {
    const colours = ['blue', 'red', 'orange', 'green', 'yellow', 'purple', 'pink', 'black', 'turquoise', 'crimson'];
    const generateDatasets: any = [];
    console.log(this.data);
    this.generateTraces();
  }

}
