### Functions used by lots of other modules. No methods are defined here - other modules import this module to overload the functions declared here.

module CommonFunctions


export allocate_params, allocate_initparams, allocate_receptive_field_norms, allocate_states, make_predictions!, get_gradient_activations!, get_gradient_parameters!, get_gradient_init_parameters!, get_u0!, change_nObs!, to_cpu!, to_gpu!

function allocate_params()
end

function allocate_initparams()
end

function allocate_receptive_field_norms()
end

function allocate_states()
end

function make_predictions!()
end

function get_gradient_activations!()
end

function get_gradient_parameters!()
end

function get_gradient_init_parameters!()
end

function get_u0!()
end

function change_nObs!()
end

function to_cpu!()
end

function to_gpu!()
end


end