import { Options } from '@angular-slider/ngx-slider';
import { Component,EventEmitter, Output } from '@angular/core';
import { NgZone } from '@angular/core';

@Component({
  selector: 'app-feature-weighting-component',
  templateUrl: './feature-weighting-component.component.html',
  styleUrls: ['./feature-weighting-component.component.css']
})
export class FeatureWeightingComponentComponent {
  sliderValues: number[] = [2, 2, 2, 2, 2];
  @Output() public rewighCluster: EventEmitter<any> = new EventEmitter<any>();

  constructor(private ngZone:NgZone){}
  sliderOptions: any = [
    {
      ticksArray: ['Not at all important', 'Less important', 'No opinion', 'Little important', 'Very Important'],
      ticksTooltip: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      translate: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      stepsArray: [
        {value: 0, legend: 'Not at all important'},
        {value: 1, legend: 'Less important'},
        {value: 2, legend: 'No opinion'},
        {value: 3, legend: 'Little important'},
        {value: 4, legend: 'Very Important'}
      ],
      showTicks: true,
      showTicksValues: true,
      animate: false
    },
    {
      ticksArray: ['Not at all important', 'Less important', 'No opinion', 'Little important', 'Very Important'],
      ticksTooltip: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      translate: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      stepsArray: [
        {value: 0, legend: 'Not at all important'},
        {value: 1, legend: 'Less important'},
        {value: 2, legend: 'No opinion'},
        {value: 3, legend: 'Little important'},
        {value: 4, legend: 'Very Important'}
      ],
      showTicks: true,
      showTicksValues: true,
      animate: false
    },
    {
      ticksArray: ['Not at all important', 'Less important', 'No opinion', 'Little important', 'Very Important'],
      ticksTooltip: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      translate: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      stepsArray: [
        {value: 0, legend: 'Not at all important'},
        {value: 1, legend: 'Less important'},
        {value: 2, legend: 'No opinion'},
        {value: 3, legend: 'Little important'},
        {value: 4, legend: 'Very Important'}
      ],
      showTicks: true,
      showTicksValues: true,
      animate: false
    },
    {
      ticksArray: ['Not at all important', 'Less important', 'No opinion', 'Little important', 'Very Important'],
      ticksTooltip: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      translate: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      stepsArray: [
        {value: 0, legend: 'Not at all important'},
        {value: 1, legend: 'Less important'},
        {value: 2, legend: 'No opinion'},
        {value: 3, legend: 'Little important'},
        {value: 4, legend: 'Very Important'}
      ],
      showTicks: true,
      showTicksValues: true,
      animate: false
    },
    {
      ticksArray: ['Not at all important', 'Less important', 'No opinion', 'Little important', 'Very Important'],
      ticksTooltip: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      translate: (value: number) => {
        return this.sliderOptions[1].ticksArray[value];
      },
      stepsArray: [
        {value: 0, legend: 'Not at all important'},
        {value: 1, legend: 'Less important'},
        {value: 2, legend: 'No opinion'},
        {value: 3, legend: 'Little important'},
        {value: 4, legend: 'Very Important'}
      ],
      showTicks: true,
      showTicksValues: true,
      animate: false
    },
  ];

  onSubmit(){
    console.log(`The slider values are ${this.sliderValues}`);
    this.ngZone.run(() => this.rewighCluster.emit(this.sliderValues));
  }
}
