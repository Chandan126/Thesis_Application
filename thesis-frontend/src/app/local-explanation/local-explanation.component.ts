import { Component,Input} from '@angular/core';
@Component({
  selector: 'app-local-explanation',
  templateUrl: './local-explanation.component.html',
  styleUrls: ['./local-explanation.component.css'],
})
export class LocalExplanationComponent {
  @Input() localExplanations: any;
  @Input() article1: any;
  @Input() article2: any;
}
