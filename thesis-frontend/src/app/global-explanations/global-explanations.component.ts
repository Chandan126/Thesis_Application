import { Component,Input } from '@angular/core';
import {DataServiceService} from '../data-service.service';


@Component({
  selector: 'app-global-explanations',
  templateUrl: './global-explanations.component.html',
  styleUrls: ['./global-explanations.component.css']
})
export class GlobalExplanationsComponent {
  @Input() clusterNumber: string;
  @Input() globalExplanations: any = [];
  importantWords: string[];
  //@Output() public clusterFeedback: EventEmitter<any> = new EventEmitter<any>();

  constructor(private dataService: DataServiceService) { 
  }

  ngOnInit(): void {
    this.importantWords = this.dataService.getQuery()?.split(" ") || [];
  }
}
