import { Component, Inject, AfterViewInit  } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  selector: 'app-facet-explanation-component',
  templateUrl: './facet-explanation-component.component.html',
  styleUrls: ['./facet-explanation-component.component.css']
})
export class FacetExplanationComponentComponent implements AfterViewInit{
  selected_article: string;
  constructor(@Inject(MAT_DIALOG_DATA) public data: any) { 
    this.selected_article = data.selected_article;
    console.log(this.selected_article);
  }

  ngAfterViewInit() {
    console.log('Message in ngAfterViewInit:', this.selected_article);
  }
}
