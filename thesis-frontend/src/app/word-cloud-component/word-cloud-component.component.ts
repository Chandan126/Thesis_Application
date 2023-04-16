import { Component, Output, Input, ChangeDetectorRef, OnChanges,EventEmitter } from '@angular/core';
import * as d3 from 'd3';
import * as d3Cloud from 'd3-cloud';
import {DataServiceService} from '../data-service.service';
import { NgZone } from '@angular/core';

@Component({
  selector: 'app-word-cloud-component',
  templateUrl: './word-cloud-component.component.html',
  styleUrls: ['./word-cloud-component.component.css']
})
export class WordCloudComponentComponent implements OnChanges {

  @Input() article: any;
  @Input() clickedBar: any;
  @Input() source: any;
  @Output() public wordClicked: EventEmitter<any> = new EventEmitter<any>();
  @Output() public clusterFeedback: EventEmitter<any> = new EventEmitter<any>();
  interestingClusters: any = [];
  notinterestingClusters: any = [];
  data: any;

  constructor(private dataService: DataServiceService, private cdr: ChangeDetectorRef, private ngZone:NgZone) {
  }

  async ngOnChanges() {
    try {
      this.data = await this.dataService.getFacetExplanation(this.source,this.clickedBar,this.article).toPromise();
      this.cdr.detectChanges();
      this.generateWordCloud(this.data.facet_words);
    } catch (error) {
      console.log(error);
    }
  }

  drawWordCloud(words: any[]) {
    const svg = d3.select('svg');
    const g = svg.append('g')
      .attr('transform', 'translate(300,150)');
  
      g.selectAll('text')
      .data(words)
      .enter().append('text')
        .style('font-size', d => d.size + 'px')
        .style('font-family', 'Impact')
        .style('fill', () => `rgb(${~~(Math.random() * 255)}, ${~~(Math.random() * 255)}, ${~~(Math.random() * 255)})`)
        .attr('text-anchor', 'middle')
        .attr('transform', d => `translate(${[d.x, d.y]})rotate(${d.rotate})`)
        .text(d => d.text)
        .on('click', d => {
            console.log(`Clicked on word "${d.target.textContent}"`);
            this.ngZone.run(() => this.wordClicked.emit(d.target.textContent));
        });
  }

  clearWordCloud() {
    d3.select('svg').selectAll('*').remove();
  }

  generateWordCloud(data: any[]) {
    this.clearWordCloud();
    const layout = d3Cloud()
    .size([600, 500])
    .words(this.data.facet_words.map((d: { text: any; }) => ({text: d.text, size: 30})))
    .padding(3)
    .rotate(() => (~~(Math.random() * 2) * 0))
    .font('Impact')
    .fontSize(d => {return d.size? d.size : 30})
    .on('end', this.drawWordCloud.bind(this));

  layout.start();
  }

  onConfirmClick(resp: any, cluster: any) {
    if(resp==='yes'){
      this.dataService.storeClusterFeedback(cluster, true);
    } else{
      this.dataService.storeClusterFeedback(cluster, false);
    }
    //console.log(`User wants to move ${this.clickedWord} from ${this.clickedBar} to ${this.selectedCluster}`)
}
}


