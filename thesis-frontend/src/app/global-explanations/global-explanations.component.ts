import { Component,Input } from '@angular/core';

@Component({
  selector: 'app-global-explanations',
  templateUrl: './global-explanations.component.html',
  styleUrls: ['./global-explanations.component.css']
})
export class GlobalExplanationsComponent {
  @Input() clusterNumber: string;
  @Input() globalExplanations: any = [];
}
