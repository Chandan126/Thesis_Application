import { Component,Input,ViewEncapsulation } from '@angular/core';
import { HighlightWordsPipe } from '../highlight-words.pipe';
@Component({
  selector: 'app-local-explanation',
  templateUrl: './local-explanation.component.html',
  styleUrls: ['./local-explanation.component.css'],
  providers: [HighlightWordsPipe],
  encapsulation: ViewEncapsulation.None
})
export class LocalExplanationComponent {
  @Input() localExplanations: any;
}
