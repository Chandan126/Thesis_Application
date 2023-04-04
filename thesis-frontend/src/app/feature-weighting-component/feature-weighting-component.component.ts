import { Options } from '@angular-slider/ngx-slider';
import { Component,EventEmitter, Output } from '@angular/core';
import { NgZone } from '@angular/core';

@Component({
  selector: 'app-feature-weighting-component',
  templateUrl: './feature-weighting-component.component.html',
  styleUrls: ['./feature-weighting-component.component.css']
})
export class FeatureWeightingComponentComponent {
  sliderValues: number[] = [1, 1, 1, 1, 1];
  @Output() public rewighCluster: EventEmitter<any> = new EventEmitter<any>();

  constructor(private ngZone:NgZone){}
  sliderOptions: Options[] = [
    { floor: -5, ceil: 5 },
    { floor: -5, ceil: 5 },
    { floor: -5, ceil: 5 },
    { floor: -5, ceil: 5 },
    { floor: -5, ceil: 5 },
  ];

  onSubmit(){
    console.log(`The slider values are ${this.sliderValues}`);
    this.ngZone.run(() => this.rewighCluster.emit(this.sliderValues));
  }
}
