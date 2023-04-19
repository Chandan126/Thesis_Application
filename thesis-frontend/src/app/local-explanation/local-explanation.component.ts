import { Component,Input,OnChanges} from '@angular/core';
@Component({
  selector: 'app-local-explanation',
  templateUrl: './local-explanation.component.html',
  styleUrls: ['./local-explanation.component.css'],
})
export class LocalExplanationComponent implements OnChanges {
  @Input() localExplanations: any;
  @Input() article1: any;
  @Input() article2: any;

  ngOnChanges(): void {
    console.log(this.article1);
    console.log(this.article2);
    console.log(this.localExplanations);
  }
  
}
